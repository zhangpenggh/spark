/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package org.apache.spark.ml.regression

import scala.collection.mutable

import breeze.linalg.{DenseVector => BDV}
import breeze.optimize.{CachedDiffFunction, LBFGS => BreezeLBFGS, LBFGSB => BreezeLBFGSB, OWLQN => BreezeOWLQN}
import breeze.stats.distributions.StudentsT
import org.apache.hadoop.fs.Path

import org.apache.spark.SparkException
import org.apache.spark.annotation.{Experimental, Since}
import org.apache.spark.internal.Logging
import org.apache.spark.ml.{PipelineStage, PredictorParams}
import org.apache.spark.ml.feature.Instance
import org.apache.spark.ml.linalg.{Vector, Vectors}
import org.apache.spark.ml.linalg.BLAS._
import org.apache.spark.ml.optim.WeightedLeastSquares
import org.apache.spark.ml.optim.aggregator.{HuberAggregator, LeastSquaresAggregator}
import org.apache.spark.ml.optim.loss.{L2Regularization, RDDLossFunction}
import org.apache.spark.ml.param.{DoubleParam, Param, ParamMap, ParamValidators}
import org.apache.spark.ml.param.shared._
import org.apache.spark.ml.util._
import org.apache.spark.ml.util.Instrumentation.instrumented
import org.apache.spark.mllib.evaluation.RegressionMetrics
import org.apache.spark.mllib.linalg.VectorImplicits._
import org.apache.spark.mllib.regression.{LinearRegressionModel => OldLinearRegressionModel}
import org.apache.spark.mllib.stat.MultivariateOnlineSummarizer
import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.{DataFrame, Dataset, Row, SparkSession}
import org.apache.spark.sql.functions._
import org.apache.spark.sql.types.{DataType, DoubleType, StructType}
import org.apache.spark.storage.StorageLevel
import org.apache.spark.util.VersionUtils.majorMinorVersion

/**
 * Params for linear regression.
 */
private[regression] trait LinearRegressionParams extends PredictorParams
    with HasRegParam with HasElasticNetParam with HasMaxIter with HasTol
    with HasFitIntercept with HasStandardization with HasWeightCol with HasSolver
    with HasAggregationDepth with HasLoss {

  import LinearRegression._

  /**
   * The solver algorithm for optimization.
   * Supported options: "l-bfgs", "normal" and "auto".
   * Default: "auto"
   *
   * @group param
   */
  @Since("1.6.0")
  final override val solver: Param[String] = new Param[String](this, "solver",
    "The solver algorithm for optimization. Supported options: " +
      s"${supportedSolvers.mkString(", ")}. (Default auto)",
    ParamValidators.inArray[String](supportedSolvers))

  /**
   * The loss function to be optimized.
   * Supported options: "squaredError" and "huber".
   * Default: "squaredError"
   *
   * @group param
   */
  @Since("2.3.0")
  final override val loss: Param[String] = new Param[String](this, "loss", "The loss function to" +
    s" be optimized. Supported options: ${supportedLosses.mkString(", ")}. (Default squaredError)",
    ParamValidators.inArray[String](supportedLosses))

  /**
   * The shape parameter to control the amount of robustness. Must be &gt; 1.0.
   * At larger values of epsilon, the huber criterion becomes more similar to least squares
   * regression; for small values of epsilon, the criterion is more similar to L1 regression.
   * Default is 1.35 to get as much robustness as possible while retaining
   * 95% statistical efficiency for normally distributed data. It matches sklearn
   * HuberRegressor and is "M" from <a href="http://statweb.stanford.edu/~owen/reports/hhu.pdf">
   * A robust hybrid of lasso and ridge regression</a>.
   * Only valid when "loss" is "huber".
   *
   * @group expertParam
   */
  @Since("2.3.0")
  final val epsilon = new DoubleParam(this, "epsilon", "The shape parameter to control the " +
    "amount of robustness. Must be > 1.0.", ParamValidators.gt(1.0))

  /** @group getExpertParam */
  @Since("2.3.0")
  def getEpsilon: Double = $(epsilon)

  override protected def validateAndTransformSchema(
      schema: StructType,
      fitting: Boolean,
      featuresDataType: DataType): StructType = {
    if ($(loss) == Huber) {
      require($(solver)!= Normal, "LinearRegression with huber loss doesn't support " +
        "normal solver, please change solver to auto or l-bfgs.")
      require($(elasticNetParam) == 0.0, "LinearRegression with huber loss only supports " +
        s"L2 regularization, but got elasticNetParam = $getElasticNetParam.")

    }
    super.validateAndTransformSchema(schema, fitting, featuresDataType)
  }
}

/**
 * Linear regression.
 *
 * The learning objective is to minimize the specified loss function, with regularization.
 * This supports two kinds of loss:
 *  - squaredError (a.k.a squared loss)
 *  - huber (a hybrid of squared error for relatively small errors and absolute error for
 *  relatively large ones, and we estimate the scale parameter from training data)
 *
 * This supports multiple types of regularization:
 *  - none (a.k.a. ordinary least squares)
 *  - L2 (ridge regression)
 *  - L1 (Lasso)
 *  - L2 + L1 (elastic net)
 *
 * The squared error objective function is:
 *
 * <blockquote>
 *   $$
 *   \begin{align}
 *   \min_{w}\frac{1}{2n}{\sum_{i=1}^n(X_{i}w - y_{i})^{2} +
 *   \lambda\left[\frac{1-\alpha}{2}{||w||_{2}}^{2} + \alpha{||w||_{1}}\right]}
 *   \end{align}
 *   $$
 * </blockquote>
 *
 * The huber objective function is:
 *
 * <blockquote>
 *   $$
 *   \begin{align}
 *   \min_{w, \sigma}\frac{1}{2n}{\sum_{i=1}^n\left(\sigma +
 *   H_m\left(\frac{X_{i}w - y_{i}}{\sigma}\right)\sigma\right) + \frac{1}{2}\lambda {||w||_2}^2}
 *   \end{align}
 *   $$
 * </blockquote>
 *
 * where
 *
 * <blockquote>
 *   $$
 *   \begin{align}
 *   H_m(z) = \begin{cases}
 *            z^2, & \text {if } |z| &lt; \epsilon, \\
 *            2\epsilon|z| - \epsilon^2, & \text{otherwise}
 *            \end{cases}
 *   \end{align}
 *   $$
 * </blockquote>
 *
 * Note: Fitting with huber loss only supports none and L2 regularization.
 */
@Since("1.3.0")
class LinearRegression @Since("1.3.0") (@Since("1.3.0") override val uid: String)
  extends Regressor[Vector, LinearRegression, LinearRegressionModel]
  with LinearRegressionParams with DefaultParamsWritable with Logging {

  import LinearRegression._

  @Since("1.4.0")
  def this() = this(Identifiable.randomUID("linReg"))

  /**
   * Set the regularization parameter.
   * Default is 0.0.
   *
   * @group setParam
   */
  @Since("1.3.0")
  def setRegParam(value: Double): this.type = set(regParam, value)
  setDefault(regParam -> 0.0)

  /**
   * Set if we should fit the intercept.
   * Default is true.
   *
   * @group setParam
   */
  @Since("1.5.0")
  def setFitIntercept(value: Boolean): this.type = set(fitIntercept, value)
  setDefault(fitIntercept -> true)

  /**
   * Whether to standardize the training features before fitting the model.
   * The coefficients of models will be always returned on the original scale,
   * so it will be transparent for users.
   * Default is true.
   *
   * @note With/without standardization, the models should be always converged
   * to the same solution when no regularization is applied. In R's GLMNET package,
   * the default behavior is true as well.
   *
   * @group setParam
   */
  @Since("1.5.0")
  def setStandardization(value: Boolean): this.type = set(standardization, value)
  setDefault(standardization -> true)

  /**
   * Set the ElasticNet mixing parameter.
   * For alpha = 0, the penalty is an L2 penalty.
   * For alpha = 1, it is an L1 penalty.
   * For alpha in (0,1), the penalty is a combination of L1 and L2.
   * Default is 0.0 which is an L2 penalty.
   *
   * Note: Fitting with huber loss only supports None and L2 regularization,
   * so throws exception if this param is non-zero value.
   *
   * @group setParam
   */
  @Since("1.4.0")
  def setElasticNetParam(value: Double): this.type = set(elasticNetParam, value)
  setDefault(elasticNetParam -> 0.0)

  /**
   * Set the maximum number of iterations.
   * Default is 100.
   *
   * @group setParam
   */
  @Since("1.3.0")
  def setMaxIter(value: Int): this.type = set(maxIter, value)
  setDefault(maxIter -> 100)

  /**
   * Set the convergence tolerance of iterations.
   * Smaller value will lead to higher accuracy with the cost of more iterations.
   * Default is 1E-6.
   *
   * @group setParam
   */
  @Since("1.4.0")
  def setTol(value: Double): this.type = set(tol, value)
  setDefault(tol -> 1E-6)

  /**
   * Whether to over-/under-sample training instances according to the given weights in weightCol.
   * If not set or empty, all instances are treated equally (weight 1.0).
   * Default is not set, so all instances have weight one.
   *
   * @group setParam
   */
  @Since("1.6.0")
  def setWeightCol(value: String): this.type = set(weightCol, value)

  /**
   * Set the solver algorithm used for optimization.
   * In case of linear regression, this can be "l-bfgs", "normal" and "auto".
   *  - "l-bfgs" denotes Limited-memory BFGS which is a limited-memory quasi-Newton
   *    optimization method.
   *  - "normal" denotes using Normal Equation as an analytical solution to the linear regression
   *    problem.  This solver is limited to `LinearRegression.MAX_FEATURES_FOR_NORMAL_SOLVER`.
   *  - "auto" (default) means that the solver algorithm is selected automatically.
   *    The Normal Equations solver will be used when possible, but this will automatically fall
   *    back to iterative optimization methods when needed.
   *
   * Note: Fitting with huber loss doesn't support normal solver,
   * so throws exception if this param was set with "normal".
   * @group setParam
   */
  @Since("1.6.0")
  def setSolver(value: String): this.type = set(solver, value)
  setDefault(solver -> Auto)

  /**
   * Suggested depth for treeAggregate (greater than or equal to 2).
   * If the dimensions of features or the number of partitions are large,
   * this param could be adjusted to a larger size.
   * Default is 2.
   *
   * @group expertSetParam
   */
  @Since("2.1.0")
  def setAggregationDepth(value: Int): this.type = set(aggregationDepth, value)
  setDefault(aggregationDepth -> 2)

  /**
   * Sets the value of param [[loss]].
   * Default is "squaredError".
   *
   * @group setParam
   */
  @Since("2.3.0")
  def setLoss(value: String): this.type = set(loss, value)
  setDefault(loss -> SquaredError)

  /**
   * Sets the value of param [[epsilon]].
   * Default is 1.35.
   *
   * @group setExpertParam
   */
  @Since("2.3.0")
  def setEpsilon(value: Double): this.type = set(epsilon, value)
  setDefault(epsilon -> 1.35)

  override protected def train(dataset: Dataset[_]): LinearRegressionModel = instrumented { instr =>
    // Extract the number of features before deciding optimization solver.
    val numFeatures = dataset.select(col($(featuresCol))).first().getAs[Vector](0).size
    val w = if (!isDefined(weightCol) || $(weightCol).isEmpty) lit(1.0) else col($(weightCol))

    val instances: RDD[Instance] = dataset.select(
      col($(labelCol)), w, col($(featuresCol))).rdd.map {
      case Row(label: Double, weight: Double, features: Vector) =>
        Instance(label, weight, features)
    }

    instr.logPipelineStage(this)
    instr.logDataset(dataset)
    instr.logParams(this, labelCol, featuresCol, weightCol, predictionCol, solver, tol,
      elasticNetParam, fitIntercept, maxIter, regParam, standardization, aggregationDepth, loss,
      epsilon)
    instr.logNumFeatures(numFeatures)

    if ($(loss) == SquaredError && (($(solver) == Auto &&
      numFeatures <= WeightedLeastSquares.MAX_NUM_FEATURES) || $(solver) == Normal)) {
      // For low dimensional data, WeightedLeastSquares is more efficient since the
      // training algorithm only requires one pass through the data. (SPARK-10668)

      val optimizer = new WeightedLeastSquares($(fitIntercept), $(regParam),
        elasticNetParam = $(elasticNetParam), $(standardization), true,
        solverType = WeightedLeastSquares.Auto, maxIter = $(maxIter), tol = $(tol))
      val model = optimizer.fit(instances, instr = OptionalInstrumentation.create(instr))
      // When it is trained by WeightedLeastSquares, training summary does not
      // attach returned model.
      val lrModel = copyValues(new LinearRegressionModel(uid, model.coefficients, model.intercept))
      val (summaryModel, predictionColName) = lrModel.findSummaryModelAndPredictionCol()
      val trainingSummary = new LinearRegressionTrainingSummary(
        summaryModel.transform(dataset),
        predictionColName,
        $(labelCol),
        $(featuresCol),
        summaryModel,
        model.diagInvAtWA.toArray,
        model.objectiveHistory)

      return lrModel.setSummary(Some(trainingSummary))
    }

    val handlePersistence = dataset.storageLevel == StorageLevel.NONE
    if (handlePersistence) instances.persist(StorageLevel.MEMORY_AND_DISK)

    val (featuresSummarizer, ySummarizer) = {
      val seqOp = (c: (MultivariateOnlineSummarizer, MultivariateOnlineSummarizer),
        instance: Instance) =>
          (c._1.add(instance.features, instance.weight),
            c._2.add(Vectors.dense(instance.label), instance.weight))

      val combOp = (c1: (MultivariateOnlineSummarizer, MultivariateOnlineSummarizer),
        c2: (MultivariateOnlineSummarizer, MultivariateOnlineSummarizer)) =>
          (c1._1.merge(c2._1), c1._2.merge(c2._2))

      instances.treeAggregate(
        (new MultivariateOnlineSummarizer, new MultivariateOnlineSummarizer)
      )(seqOp, combOp, $(aggregationDepth))
    }

    val yMean = ySummarizer.mean(0)
    val rawYStd = math.sqrt(ySummarizer.variance(0))

    instr.logNumExamples(ySummarizer.count)
    instr.logNamedValue(Instrumentation.loggerTags.meanOfLabels, yMean)
    instr.logNamedValue(Instrumentation.loggerTags.varianceOfLabels, rawYStd)

    if (rawYStd == 0.0) {
      if ($(fitIntercept) || yMean == 0.0) {
        // If the rawYStd==0 and fitIntercept==true, then the intercept is yMean with
        // zero coefficient; as a result, training is not needed.
        // Also, if yMean==0 and rawYStd==0, all the coefficients are zero regardless of
        // the fitIntercept.
        if (yMean == 0.0) {
          instr.logWarning(s"Mean and standard deviation of the label are zero, so the " +
            s"coefficients and the intercept will all be zero; as a result, training is not " +
            s"needed.")
        } else {
          instr.logWarning(s"The standard deviation of the label is zero, so the coefficients " +
            s"will be zeros and the intercept will be the mean of the label; as a result, " +
            s"training is not needed.")
        }
        if (handlePersistence) instances.unpersist()
        val coefficients = Vectors.sparse(numFeatures, Seq.empty)
        val intercept = yMean

        val model = copyValues(new LinearRegressionModel(uid, coefficients, intercept))
        // Handle possible missing or invalid prediction columns
        val (summaryModel, predictionColName) = model.findSummaryModelAndPredictionCol()

        val trainingSummary = new LinearRegressionTrainingSummary(
          summaryModel.transform(dataset),
          predictionColName,
          $(labelCol),
          $(featuresCol),
          model,
          Array(0D),
          Array(0D))

        return model.setSummary(Some(trainingSummary))
      } else {
        require($(regParam) == 0.0, "The standard deviation of the label is zero. " +
          "Model cannot be regularized.")
        instr.logWarning(s"The standard deviation of the label is zero. " +
          "Consider setting fitIntercept=true.")
      }
    }

    // if y is constant (rawYStd is zero), then y cannot be scaled. In this case
    // setting yStd=abs(yMean) ensures that y is not scaled anymore in l-bfgs algorithm.
    val yStd = if (rawYStd > 0) rawYStd else math.abs(yMean)
    val featuresMean = featuresSummarizer.mean.toArray
    val featuresStd = featuresSummarizer.variance.toArray.map(math.sqrt)
    val bcFeaturesMean = instances.context.broadcast(featuresMean)
    val bcFeaturesStd = instances.context.broadcast(featuresStd)

    if (!$(fitIntercept) && (0 until numFeatures).exists { i =>
      featuresStd(i) == 0.0 && featuresMean(i) != 0.0 }) {
      instr.logWarning("Fitting LinearRegressionModel without intercept on dataset with " +
        "constant nonzero column, Spark MLlib outputs zero coefficients for constant nonzero " +
        "columns. This behavior is the same as R glmnet but different from LIBSVM.")
    }

    // Since we implicitly do the feature scaling when we compute the cost function
    // to improve the convergence, the effective regParam will be changed.
    val effectiveRegParam = $(loss) match {
      case SquaredError => $(regParam) / yStd
      case Huber => $(regParam)
    }
    val effectiveL1RegParam = $(elasticNetParam) * effectiveRegParam
    val effectiveL2RegParam = (1.0 - $(elasticNetParam)) * effectiveRegParam

    val getFeaturesStd = (j: Int) => if (j >= 0 && j < numFeatures) featuresStd(j) else 0.0
    val regularization = if (effectiveL2RegParam != 0.0) {
      val shouldApply = (idx: Int) => idx >= 0 && idx < numFeatures
      Some(new L2Regularization(effectiveL2RegParam, shouldApply,
        if ($(standardization)) None else Some(getFeaturesStd)))
    } else {
      None
    }

    val costFun = $(loss) match {
      case SquaredError =>
        val getAggregatorFunc = new LeastSquaresAggregator(yStd, yMean, $(fitIntercept),
          bcFeaturesStd, bcFeaturesMean)(_)
        new RDDLossFunction(instances, getAggregatorFunc, regularization, $(aggregationDepth))
      case Huber =>
        val getAggregatorFunc = new HuberAggregator($(fitIntercept), $(epsilon), bcFeaturesStd)(_)
        new RDDLossFunction(instances, getAggregatorFunc, regularization, $(aggregationDepth))
    }

    val optimizer = $(loss) match {
      case SquaredError =>
        if ($(elasticNetParam) == 0.0 || effectiveRegParam == 0.0) {
          new BreezeLBFGS[BDV[Double]]($(maxIter), 10, $(tol))
        } else {
          val standardizationParam = $(standardization)
          def effectiveL1RegFun = (index: Int) => {
            if (standardizationParam) {
              effectiveL1RegParam
            } else {
              // If `standardization` is false, we still standardize the data
              // to improve the rate of convergence; as a result, we have to
              // perform this reverse standardization by penalizing each component
              // differently to get effectively the same objective function when
              // the training dataset is not standardized.
              if (featuresStd(index) != 0.0) effectiveL1RegParam / featuresStd(index) else 0.0
            }
          }
          new BreezeOWLQN[Int, BDV[Double]]($(maxIter), 10, effectiveL1RegFun, $(tol))
        }
      case Huber =>
        val dim = if ($(fitIntercept)) numFeatures + 2 else numFeatures + 1
        val lowerBounds = BDV[Double](Array.fill(dim)(Double.MinValue))
        // Optimize huber loss in space "\sigma > 0"
        lowerBounds(dim - 1) = Double.MinPositiveValue
        val upperBounds = BDV[Double](Array.fill(dim)(Double.MaxValue))
        new BreezeLBFGSB(lowerBounds, upperBounds, $(maxIter), 10, $(tol))
    }

    val initialValues = $(loss) match {
      case SquaredError =>
        Vectors.zeros(numFeatures)
      case Huber =>
        val dim = if ($(fitIntercept)) numFeatures + 2 else numFeatures + 1
        Vectors.dense(Array.fill(dim)(1.0))
    }

    val states = optimizer.iterations(new CachedDiffFunction(costFun),
      initialValues.asBreeze.toDenseVector)

    val (coefficients, intercept, scale, objectiveHistory) = {
      /*
         Note that in Linear Regression, the objective history (loss + regularization) returned
         from optimizer is computed in the scaled space given by the following formula.
         <blockquote>
            $$
            L &= 1/2n||\sum_i w_i(x_i - \bar{x_i}) / \hat{x_i} - (y - \bar{y}) / \hat{y}||^2
                 + regTerms \\
            $$
         </blockquote>
       */
      val arrayBuilder = mutable.ArrayBuilder.make[Double]
      var state: optimizer.State = null
      while (states.hasNext) {
        state = states.next()
        arrayBuilder += state.adjustedValue
      }
      if (state == null) {
        val msg = s"${optimizer.getClass.getName} failed."
        instr.logError(msg)
        throw new SparkException(msg)
      }

      bcFeaturesMean.destroy(blocking = false)
      bcFeaturesStd.destroy(blocking = false)

      val parameters = state.x.toArray.clone()

      /*
         The coefficients are trained in the scaled space; we're converting them back to
         the original space.
       */
      val rawCoefficients: Array[Double] = $(loss) match {
        case SquaredError => parameters
        case Huber => parameters.slice(0, numFeatures)
      }

      var i = 0
      val len = rawCoefficients.length
      val multiplier = $(loss) match {
        case SquaredError => yStd
        case Huber => 1.0
      }
      while (i < len) {
        rawCoefficients(i) *= { if (featuresStd(i) != 0.0) multiplier / featuresStd(i) else 0.0 }
        i += 1
      }

      val interceptValue: Double = if ($(fitIntercept)) {
        $(loss) match {
          case SquaredError =>
            /*
            The intercept of squared error in R's GLMNET is computed using closed form
            after the coefficients are converged. See the following discussion for detail.
            http://stats.stackexchange.com/questions/13617/how-is-the-intercept-computed-in-glmnet
            */
            yMean - dot(Vectors.dense(rawCoefficients), Vectors.dense(featuresMean))
          case Huber => parameters(numFeatures)
        }
      } else {
        0.0
      }

      val scaleValue: Double = $(loss) match {
        case SquaredError => 1.0
        case Huber => parameters.last
      }

      (Vectors.dense(rawCoefficients).compressed, interceptValue, scaleValue, arrayBuilder.result())
    }

    if (handlePersistence) instances.unpersist()

    val model = copyValues(new LinearRegressionModel(uid, coefficients, intercept, scale))
    // Handle possible missing or invalid prediction columns
    val (summaryModel, predictionColName) = model.findSummaryModelAndPredictionCol()

    val trainingSummary = new LinearRegressionTrainingSummary(
      summaryModel.transform(dataset),
      predictionColName,
      $(labelCol),
      $(featuresCol),
      model,
      Array(0D),
      objectiveHistory)

    model.setSummary(Some(trainingSummary))
  }

  @Since("1.4.0")
  override def copy(extra: ParamMap): LinearRegression = defaultCopy(extra)
}

@Since("1.6.0")
object LinearRegression extends DefaultParamsReadable[LinearRegression] {

  @Since("1.6.0")
  override def load(path: String): LinearRegression = super.load(path)

  /**
   * When using `LinearRegression.solver` == "normal", the solver must limit the number of
   * features to at most this number.  The entire covariance matrix X^T^X will be collected
   * to the driver. This limit helps prevent memory overflow errors.
   */
  @Since("2.1.0")
  val MAX_FEATURES_FOR_NORMAL_SOLVER: Int = WeightedLeastSquares.MAX_NUM_FEATURES

  /** String name for "auto". */
  private[regression] val Auto = "auto"

  /** String name for "normal". */
  private[regression] val Normal = "normal"

  /** String name for "l-bfgs". */
  private[regression] val LBFGS = "l-bfgs"

  /** Set of solvers that LinearRegression supports. */
  private[regression] val supportedSolvers = Array(Auto, Normal, LBFGS)

  /** String name for "squaredError". */
  private[regression] val SquaredError = "squaredError"

  /** String name for "huber". */
  private[regression] val Huber = "huber"

  /** Set of loss function names that LinearRegression supports. */
  private[regression] val supportedLosses = Array(SquaredError, Huber)
}

/**
 * Model produced by [[LinearRegression]].
 */
@Since("1.3.0")
class LinearRegressionModel private[ml] (
    @Since("1.4.0") override val uid: String,
    @Since("2.0.0") val coefficients: Vector,
    @Since("1.3.0") val intercept: Double,
    @Since("2.3.0") val scale: Double)
  extends RegressionModel[Vector, LinearRegressionModel]
  with LinearRegressionParams with GeneralMLWritable {

  private[ml] def this(uid: String, coefficients: Vector, intercept: Double) =
    this(uid, coefficients, intercept, 1.0)

  private var trainingSummary: Option[LinearRegressionTrainingSummary] = None

  override val numFeatures: Int = coefficients.size

  /**
   * Gets summary (e.g. residuals, mse, r-squared ) of model on training set. An exception is
   * thrown if `trainingSummary == None`.
   */
  @Since("1.5.0")
  def summary: LinearRegressionTrainingSummary = trainingSummary.getOrElse {
    throw new SparkException("No training summary available for this LinearRegressionModel")
  }

  private[regression]
  def setSummary(summary: Option[LinearRegressionTrainingSummary]): this.type = {
    this.trainingSummary = summary
    this
  }

  /** Indicates whether a training summary exists for this model instance. */
  @Since("1.5.0")
  def hasSummary: Boolean = trainingSummary.isDefined

  /**
   * Evaluates the model on a test dataset.
   *
   * @param dataset Test dataset to evaluate model on.
   */
  @Since("2.0.0")
  def evaluate(dataset: Dataset[_]): LinearRegressionSummary = {
    // Handle possible missing or invalid prediction columns
    val (summaryModel, predictionColName) = findSummaryModelAndPredictionCol()
    new LinearRegressionSummary(summaryModel.transform(dataset), predictionColName,
      $(labelCol), $(featuresCol), summaryModel, Array(0D))
  }

  /**
   * If the prediction column is set returns the current model and prediction column,
   * otherwise generates a new column and sets it as the prediction column on a new copy
   * of the current model.
   */
  private[regression] def findSummaryModelAndPredictionCol(): (LinearRegressionModel, String) = {
    $(predictionCol) match {
      case "" =>
        val predictionColName = "prediction_" + java.util.UUID.randomUUID.toString
        (copy(ParamMap.empty).setPredictionCol(predictionColName), predictionColName)
      case p => (this, p)
    }
  }


  override def predict(features: Vector): Double = {
    dot(features, coefficients) + intercept
  }

  @Since("1.4.0")
  override def copy(extra: ParamMap): LinearRegressionModel = {
    val newModel = copyValues(new LinearRegressionModel(uid, coefficients, intercept), extra)
    newModel.setSummary(trainingSummary).setParent(parent)
  }

  /**
   * Returns a [[org.apache.spark.ml.util.GeneralMLWriter]] instance for this ML instance.
   *
   * For [[LinearRegressionModel]], this does NOT currently save the training [[summary]].
   * An option to save [[summary]] may be added in the future.
   *
   * This also does not save the [[parent]] currently.
   */
  @Since("1.6.0")
  override def write: GeneralMLWriter = new GeneralMLWriter(this)
}

/** A writer for LinearRegression that handles the "internal" (or default) format */
private class InternalLinearRegressionModelWriter
  extends MLWriterFormat with MLFormatRegister {

  override def format(): String = "internal"
  override def stageName(): String = "org.apache.spark.ml.regression.LinearRegressionModel"

  private case class Data(intercept: Double, coefficients: Vector, scale: Double)

  override def write(path: String, sparkSession: SparkSession,
    optionMap: mutable.Map[String, String], stage: PipelineStage): Unit = {
    val instance = stage.asInstanceOf[LinearRegressionModel]
    val sc = sparkSession.sparkContext
    // Save metadata and Params
    DefaultParamsWriter.saveMetadata(instance, path, sc)
    // Save model data: intercept, coefficients, scale
    val data = Data(instance.intercept, instance.coefficients, instance.scale)
    val dataPath = new Path(path, "data").toString
    sparkSession.createDataFrame(Seq(data)).repartition(1).write.parquet(dataPath)
  }
}

/** A writer for LinearRegression that handles the "pmml" format */
private class PMMLLinearRegressionModelWriter
  extends MLWriterFormat with MLFormatRegister {

  override def format(): String = "pmml"

  override def stageName(): String = "org.apache.spark.ml.regression.LinearRegressionModel"

  private case class Data(intercept: Double, coefficients: Vector)

  override def write(path: String, sparkSession: SparkSession,
    optionMap: mutable.Map[String, String], stage: PipelineStage): Unit = {
    val sc = sparkSession.sparkContext
    // Construct the MLLib model which knows how to write to PMML.
    val instance = stage.asInstanceOf[LinearRegressionModel]
    val oldModel = new OldLinearRegressionModel(instance.coefficients, instance.intercept)
    // Save PMML
    oldModel.toPMML(sc, path)
  }
}

@Since("1.6.0")
object LinearRegressionModel extends MLReadable[LinearRegressionModel] {

  @Since("1.6.0")
  override def read: MLReader[LinearRegressionModel] = new LinearRegressionModelReader

  @Since("1.6.0")
  override def load(path: String): LinearRegressionModel = super.load(path)

  private class LinearRegressionModelReader extends MLReader[LinearRegressionModel] {

    /** Checked against metadata when loading model */
    private val className = classOf[LinearRegressionModel].getName

    override def load(path: String): LinearRegressionModel = {
      val metadata = DefaultParamsReader.loadMetadata(path, sc, className)

      val dataPath = new Path(path, "data").toString
      val data = sparkSession.read.format("parquet").load(dataPath)
      val (majorVersion, minorVersion) = majorMinorVersion(metadata.sparkVersion)
      val model = if (majorVersion < 2 || (majorVersion == 2 && minorVersion <= 2)) {
        // Spark 2.2 and before
        val Row(intercept: Double, coefficients: Vector) =
          MLUtils.convertVectorColumnsToML(data, "coefficients")
            .select("intercept", "coefficients")
            .head()
        new LinearRegressionModel(metadata.uid, coefficients, intercept)
      } else {
        // Spark 2.3 and later
        val Row(intercept: Double, coefficients: Vector, scale: Double) =
          data.select("intercept", "coefficients", "scale").head()
        new LinearRegressionModel(metadata.uid, coefficients, intercept, scale)
      }

      metadata.getAndSetParams(model)
      model
    }
  }
}

/**
 * :: Experimental ::
 * Linear regression training results. Currently, the training summary ignores the
 * training weights except for the objective trace.
 *
 * @param predictions predictions output by the model's `transform` method.
 * @param objectiveHistory objective function (scaled loss + regularization) at each iteration.
 */
@Since("1.5.0")
@Experimental
class LinearRegressionTrainingSummary private[regression] (
    predictions: DataFrame,
    predictionCol: String,
    labelCol: String,
    featuresCol: String,
    model: LinearRegressionModel,
    diagInvAtWA: Array[Double],
    val objectiveHistory: Array[Double])
  extends LinearRegressionSummary(
    predictions,
    predictionCol,
    labelCol,
    featuresCol,
    model,
    diagInvAtWA) {

  /**
   * Number of training iterations until termination
   *
   * This value is only available when using the "l-bfgs" solver.
   *
   * @see `LinearRegression.solver`
   */
  @Since("1.5.0")
  val totalIterations = objectiveHistory.length

}

/**
 * :: Experimental ::
 * Linear regression results evaluated on a dataset.
 *
 * @param predictions predictions output by the model's `transform` method.
 * @param predictionCol Field in "predictions" which gives the predicted value of the label at
 *                      each instance.
 * @param labelCol Field in "predictions" which gives the true label of each instance.
 * @param featuresCol Field in "predictions" which gives the features of each instance as a vector.
 */
@Since("1.5.0")
@Experimental
class LinearRegressionSummary private[regression] (
    @transient val predictions: DataFrame,
    val predictionCol: String,
    val labelCol: String,
    val featuresCol: String,
    private val privateModel: LinearRegressionModel,
    private val diagInvAtWA: Array[Double]) extends Serializable {

  @transient private val metrics = new RegressionMetrics(
    predictions
      .select(col(predictionCol), col(labelCol).cast(DoubleType))
      .rdd
      .map { case Row(pred: Double, label: Double) => (pred, label) },
    !privateModel.getFitIntercept)

  /**
   * Returns the explained variance regression score.
   * explainedVariance = 1 - variance(y - \hat{y}) / variance(y)
   * Reference: <a href="http://en.wikipedia.org/wiki/Explained_variation">
   * Wikipedia explain variation</a>
   *
   * @note This ignores instance weights (setting all to 1.0) from `LinearRegression.weightCol`.
   * This will change in later Spark versions.
   */
  @Since("1.5.0")
  val explainedVariance: Double = metrics.explainedVariance

  /**
   * Returns the mean absolute error, which is a risk function corresponding to the
   * expected value of the absolute error loss or l1-norm loss.
   *
   * @note This ignores instance weights (setting all to 1.0) from `LinearRegression.weightCol`.
   * This will change in later Spark versions.
   */
  @Since("1.5.0")
  val meanAbsoluteError: Double = metrics.meanAbsoluteError

  /**
   * Returns the mean squared error, which is a risk function corresponding to the
   * expected value of the squared error loss or quadratic loss.
   *
   * @note This ignores instance weights (setting all to 1.0) from `LinearRegression.weightCol`.
   * This will change in later Spark versions.
   */
  @Since("1.5.0")
  val meanSquaredError: Double = metrics.meanSquaredError

  /**
   * Returns the root mean squared error, which is defined as the square root of
   * the mean squared error.
   *
   * @note This ignores instance weights (setting all to 1.0) from `LinearRegression.weightCol`.
   * This will change in later Spark versions.
   */
  @Since("1.5.0")
  val rootMeanSquaredError: Double = metrics.rootMeanSquaredError

  /**
   * Returns R^2^, the coefficient of determination.
   * Reference: <a href="http://en.wikipedia.org/wiki/Coefficient_of_determination">
   * Wikipedia coefficient of determination</a>
   *
   * @note This ignores instance weights (setting all to 1.0) from `LinearRegression.weightCol`.
   * This will change in later Spark versions.
   */
  @Since("1.5.0")
  val r2: Double = metrics.r2

  /**
   * Returns Adjusted R^2^, the adjusted coefficient of determination.
   * Reference: <a href="https://en.wikipedia.org/wiki/Coefficient_of_determination#Adjusted_R2">
   * Wikipedia coefficient of determination</a>
   *
   * @note This ignores instance weights (setting all to 1.0) from `LinearRegression.weightCol`.
   * This will change in later Spark versions.
   */
  @Since("2.3.0")
  val r2adj: Double = {
    val interceptDOF = if (privateModel.getFitIntercept) 1 else 0
    1 - (1 - r2) * (numInstances - interceptDOF) /
      (numInstances - privateModel.coefficients.size - interceptDOF)
  }

  /** Residuals (label - predicted value) */
  @Since("1.5.0")
  @transient lazy val residuals: DataFrame = {
    val t = udf { (pred: Double, label: Double) => label - pred }
    predictions.select(t(col(predictionCol), col(labelCol)).as("residuals"))
  }

  /** Number of instances in DataFrame predictions */
  lazy val numInstances: Long = predictions.count()

  /** Degrees of freedom */
  @Since("2.2.0")
  val degreesOfFreedom: Long = if (privateModel.getFitIntercept) {
    numInstances - privateModel.coefficients.size - 1
  } else {
    numInstances - privateModel.coefficients.size
  }

  /**
   * The weighted residuals, the usual residuals rescaled by
   * the square root of the instance weights.
   */
  lazy val devianceResiduals: Array[Double] = {
    val weighted =
      if (!privateModel.isDefined(privateModel.weightCol) || privateModel.getWeightCol.isEmpty) {
        lit(1.0)
      } else {
        sqrt(col(privateModel.getWeightCol))
      }
    val dr = predictions
      .select(col(privateModel.getLabelCol).minus(col(privateModel.getPredictionCol))
        .multiply(weighted).as("weightedResiduals"))
      .select(min(col("weightedResiduals")).as("min"), max(col("weightedResiduals")).as("max"))
      .first()
    Array(dr.getDouble(0), dr.getDouble(1))
  }

  /**
   * Standard error of estimated coefficients and intercept.
   * This value is only available when using the "normal" solver.
   *
   * If `LinearRegression.fitIntercept` is set to true,
   * then the last element returned corresponds to the intercept.
   *
   * @see `LinearRegression.solver`
   */
  lazy val coefficientStandardErrors: Array[Double] = {
    if (diagInvAtWA.length == 1 && diagInvAtWA(0) == 0) {
      throw new UnsupportedOperationException(
        "No Std. Error of coefficients available for this LinearRegressionModel")
    } else {
      val rss =
        if (!privateModel.isDefined(privateModel.weightCol) || privateModel.getWeightCol.isEmpty) {
          meanSquaredError * numInstances
        } else {
          val t = udf { (pred: Double, label: Double, weight: Double) =>
            math.pow(label - pred, 2.0) * weight }
          predictions.select(t(col(privateModel.getPredictionCol), col(privateModel.getLabelCol),
            col(privateModel.getWeightCol)).as("wse")).agg(sum(col("wse"))).first().getDouble(0)
        }
      val sigma2 = rss / degreesOfFreedom
      diagInvAtWA.map(_ * sigma2).map(math.sqrt)
    }
  }

  /**
   * T-statistic of estimated coefficients and intercept.
   * This value is only available when using the "normal" solver.
   *
   * If `LinearRegression.fitIntercept` is set to true,
   * then the last element returned corresponds to the intercept.
   *
   * @see `LinearRegression.solver`
   */
  lazy val tValues: Array[Double] = {
    if (diagInvAtWA.length == 1 && diagInvAtWA(0) == 0) {
      throw new UnsupportedOperationException(
        "No t-statistic available for this LinearRegressionModel")
    } else {
      val estimate = if (privateModel.getFitIntercept) {
        Array.concat(privateModel.coefficients.toArray, Array(privateModel.intercept))
      } else {
        privateModel.coefficients.toArray
      }
      estimate.zip(coefficientStandardErrors).map { x => x._1 / x._2 }
    }
  }

  /**
   * Two-sided p-value of estimated coefficients and intercept.
   * This value is only available when using the "normal" solver.
   *
   * If `LinearRegression.fitIntercept` is set to true,
   * then the last element returned corresponds to the intercept.
   *
   * @see `LinearRegression.solver`
   */
  lazy val pValues: Array[Double] = {
    if (diagInvAtWA.length == 1 && diagInvAtWA(0) == 0) {
      throw new UnsupportedOperationException(
        "No p-value available for this LinearRegressionModel")
    } else {
      tValues.map { x => 2.0 * (1.0 - StudentsT(degreesOfFreedom.toDouble).cdf(math.abs(x))) }
    }
  }

}

<<<<<<< HEAD
/**
 * LeastSquaresAggregator computes the gradient and loss for a Least-squared loss function,
 * as used in linear regression for samples in sparse or dense vector in an online fashion.
 *
 * Two LeastSquaresAggregator can be merged together to have a summary of loss and gradient of
 * the corresponding joint dataset.
 *
 * For improving the convergence rate during the optimization process, and also preventing against
 * features with very large variances exerting an overly large influence during model training,
 * package like R's GLMNET performs the scaling to unit variance and removing the mean to reduce
 * the condition number, and then trains the model in scaled space but returns the coefficients in
 * the original scale. See page 9 in http://cran.r-project.org/web/packages/glmnet/glmnet.pdf
 *
 * However, we don't want to apply the `StandardScaler` on the training dataset, and then cache
 * the standardized dataset since it will create a lot of overhead. As a result, we perform the
 * scaling implicitly when we compute the objective function. The following is the mathematical
 * derivation.
 *
 * Note that we don't deal with intercept by adding bias here, because the intercept
 * can be computed using closed form after the coefficients are converged.
 * See this discussion for detail.
 * http://stats.stackexchange.com/questions/13617/how-is-the-intercept-computed-in-glmnet
 *
 * When training with intercept enabled,
 * The objective function in the scaled space is given by
 *
 * <blockquote>
 *    $$
 *    L = 1/2n ||\sum_i w_i(x_i - \bar{x_i}) / \hat{x_i} - (y - \bar{y}) / \hat{y}||^2,
 *    $$
 * </blockquote>
 *
 * where $\bar{x_i}$ is the mean of $x_i$, $\hat{x_i}$ is the standard deviation of $x_i$,
 * $\bar{y}$ is the mean of label, and $\hat{y}$ is the standard deviation of label.
 *
 * If we fitting the intercept disabled (that is forced through 0.0),
 * we can use the same equation except we set $\bar{y}$ and $\bar{x_i}$ to 0 instead
 * of the respective means.
 *
 * This can be rewritten as
 *
 * <blockquote>
 *    $$
 *    \begin{align}
 *     L &= 1/2n ||\sum_i (w_i/\hat{x_i})x_i - \sum_i (w_i/\hat{x_i})\bar{x_i} - y / \hat{y}
 *          + \bar{y} / \hat{y}||^2 \\
 *       &= 1/2n ||\sum_i w_i^\prime x_i - y / \hat{y} + offset||^2 = 1/2n diff^2
 *    \end{align}
 *    $$
 * </blockquote>
 *
 * where $w_i^\prime$ is the effective coefficients defined by $w_i/\hat{x_i}$, offset is
 *
 * <blockquote>
 *    $$
 *    - \sum_i (w_i/\hat{x_i})\bar{x_i} + \bar{y} / \hat{y}.
 *    $$
 * </blockquote>
 *
 * and diff is
 *
 * <blockquote>
 *    $$
 *    \sum_i w_i^\prime x_i - y / \hat{y} + offset
 *    $$
 * </blockquote>
 *
 * Note that the effective coefficients and offset don't depend on training dataset,
 * so they can be precomputed.
 *
 * Now, the first derivative of the objective function in scaled space is
 *
 * <blockquote>
 *    $$
 *    \frac{\partial L}{\partial w_i} = diff/N (x_i - \bar{x_i}) / \hat{x_i}
 *    $$
 * </blockquote>
 *
 * However, $(x_i - \bar{x_i})$ will densify the computation, so it's not
 * an ideal formula when the training dataset is sparse format.
 *
 * This can be addressed by adding the dense $\bar{x_i} / \hat{x_i}$ terms
 * in the end by keeping the sum of diff. The first derivative of total
 * objective function from all the samples is
 *
 *
 * <blockquote>
 *    $$
 *    \begin{align}
 *       \frac{\partial L}{\partial w_i} &=
 *           1/N \sum_j diff_j (x_{ij} - \bar{x_i}) / \hat{x_i} \\
 *         &= 1/N ((\sum_j diff_j x_{ij} / \hat{x_i}) - diffSum \bar{x_i} / \hat{x_i}) \\
 *         &= 1/N ((\sum_j diff_j x_{ij} / \hat{x_i}) + correction_i)
 *    \end{align}
 *    $$
 * </blockquote>
 *
 * where $correction_i = - diffSum \bar{x_i} / \hat{x_i}$
 *
 * A simple math can show that diffSum is actually zero, so we don't even
 * need to add the correction terms in the end. From the definition of diff,
 *
 * <blockquote>
 *    $$
 *    \begin{align}
 *       diffSum &= \sum_j (\sum_i w_i(x_{ij} - \bar{x_i})
 *                    / \hat{x_i} - (y_j - \bar{y}) / \hat{y}) \\
 *         &= N * (\sum_i w_i(\bar{x_i} - \bar{x_i}) / \hat{x_i} - (\bar{y} - \bar{y}) / \hat{y}) \\
 *         &= 0
 *    \end{align}
 *    $$
 * </blockquote>
 *
 * As a result, the first derivative of the total objective function only depends on
 * the training dataset, which can be easily computed in distributed fashion, and is
 * sparse format friendly.
 *
 * <blockquote>
 *    $$
 *    \frac{\partial L}{\partial w_i} = 1/N ((\sum_j diff_j x_{ij} / \hat{x_i})
 *    $$
 * </blockquote>
 *
 * @param bcCoefficients The broadcast coefficients corresponding to the features.
 * @param labelStd The standard deviation value of the label.
 * @param labelMean The mean value of the label.
 * @param fitIntercept Whether to fit an intercept term.
 * @param bcFeaturesStd The broadcast standard deviation values of the features.
 * @param bcFeaturesMean The broadcast mean values of the features.
 */
private class LeastSquaresAggregator(
    bcCoefficients: Broadcast[Vector],
    labelStd: Double,
    labelMean: Double,
    fitIntercept: Boolean,
    bcFeaturesStd: Broadcast[Array[Double]],
    bcFeaturesMean: Broadcast[Array[Double]]) extends Serializable {

  private var totalCnt: Long = 0L
  private var weightSum: Double = 0.0
  private var lossSum = 0.0

  private val dim = bcCoefficients.value.size
  // make transient so we do not serialize between aggregation stages
  @transient private lazy val featuresStd = bcFeaturesStd.value
  @transient private lazy val effectiveCoefAndOffset = {
    val coefficientsArray = bcCoefficients.value.toArray.clone()
    val featuresMean = bcFeaturesMean.value
    var sum = 0.0
    var i = 0
    val len = coefficientsArray.length
    while (i < len) {
      if (featuresStd(i) != 0.0) {
        coefficientsArray(i) /=  featuresStd(i)
        sum += coefficientsArray(i) * featuresMean(i)
      } else {
        coefficientsArray(i) = 0.0
      }
      i += 1
    }
    val offset = if (fitIntercept) labelMean / labelStd - sum else 0.0
    (Vectors.dense(coefficientsArray), offset)
  }
  // do not use tuple assignment above because it will circumvent the @transient tag
  @transient private lazy val effectiveCoefficientsVector = effectiveCoefAndOffset._1
  @transient private lazy val offset = effectiveCoefAndOffset._2

  private lazy val gradientSumArray = Array.ofDim[Double](dim)

  /**
   * Add a new training instance to this LeastSquaresAggregator, and update the loss and gradient
   * of the objective function.
   *
   * @param instance The instance of data point to be added.
   * @return This LeastSquaresAggregator object.
   */
  def add(instance: Instance): this.type = {
    instance match { case Instance(label, weight, features) =>

      if (weight == 0.0) return this

      val diff = dot(features, effectiveCoefficientsVector) - label / labelStd + offset

      if (diff != 0) {
        val localGradientSumArray = gradientSumArray
        val localFeaturesStd = featuresStd
        features.foreachActive { (index, value) =>
          if (localFeaturesStd(index) != 0.0 && value != 0.0) {
            localGradientSumArray(index) += weight * diff * value / localFeaturesStd(index)
          }
        }
        lossSum += weight * diff * diff / 2.0
      }

      totalCnt += 1
      weightSum += weight
      this
    }
  }

  /**
   * Merge another LeastSquaresAggregator, and update the loss and gradient
   * of the objective function.
   * (Note that it's in place merging; as a result, `this` object will be modified.)
   *
   * @param other The other LeastSquaresAggregator to be merged.
   * @return This LeastSquaresAggregator object.
   */
  def merge(other: LeastSquaresAggregator): this.type = {

    if (other.weightSum != 0) {
      totalCnt += other.totalCnt
      weightSum += other.weightSum
      lossSum += other.lossSum

      var i = 0
      val localThisGradientSumArray = this.gradientSumArray
      val localOtherGradientSumArray = other.gradientSumArray
      while (i < dim) {
        localThisGradientSumArray(i) += localOtherGradientSumArray(i)
        i += 1
      }
    }
    this
  }

  def count: Long = totalCnt

  def loss: Double = {
    require(weightSum > 0.0, s"The effective number of instances should be " +
      s"greater than 0.0, but $weightSum.")
    lossSum / weightSum
  }

  def gradient: Vector = {
    require(weightSum > 0.0, s"The effective number of instances should be " +
      s"greater than 0.0, but $weightSum.")
    val result = Vectors.dense(gradientSumArray.clone())
    scal(1.0 / weightSum, result)
    result
  }
}

/**
 * LeastSquaresCostFun implements Breeze's DiffFunction[T] for Least Squares cost.
 * It returns the loss and gradient with L2 regularization at a particular point (coefficients).
 * It's used in Breeze's convex optimization routines.
 */
private class LeastSquaresCostFun(
    instances: RDD[Instance],
    labelStd: Double,
    labelMean: Double,
    fitIntercept: Boolean,
    standardization: Boolean,
    bcFeaturesStd: Broadcast[Array[Double]],
    bcFeaturesMean: Broadcast[Array[Double]],
    effectiveL2regParam: Double,
    aggregationDepth: Int) extends DiffFunction[BDV[Double]] {

  override def calculate(coefficients: BDV[Double]): (Double, BDV[Double]) = {
    val coeffs = Vectors.fromBreeze(coefficients)
    val bcCoeffs = instances.context.broadcast(coeffs)
    val localFeaturesStd = bcFeaturesStd.value

    val leastSquaresAggregator = {
      val seqOp = (c: LeastSquaresAggregator, instance: Instance) => c.add(instance)
      val combOp = (c1: LeastSquaresAggregator, c2: LeastSquaresAggregator) => c1.merge(c2)

      instances.treeAggregate(
        new LeastSquaresAggregator(bcCoeffs, labelStd, labelMean, fitIntercept, bcFeaturesStd,
          bcFeaturesMean))(seqOp, combOp, aggregationDepth)
    }

    val totalGradientArray = leastSquaresAggregator.gradient.toArray
    bcCoeffs.destroy(blocking = false)

    val regVal = if (effectiveL2regParam == 0.0) {
      0.0
    } else {
      var sum = 0.0
      coeffs.foreachActive { (index, value) =>
        // The following code will compute the loss of the regularization; also
        // the gradient of the regularization, and add back to totalGradientArray.
        sum += {
          if (standardization) {
            totalGradientArray(index) += effectiveL2regParam * value
            value * value
          } else {
            if (localFeaturesStd(index) != 0.0) {
              // If `standardization` is false, we still standardize the data
              // to improve the rate of convergence; as a result, we have to
              // perform this reverse standardization by penalizing each component
              // differently to get effectively the same objective function when
              // the training dataset is not standardized.
              val temp = value / (localFeaturesStd(index) * localFeaturesStd(index))
              totalGradientArray(index) += effectiveL2regParam * temp
              value * temp
            } else {
              0.0
            }
          }
        }
      }
      0.5 * effectiveL2regParam * sum
    }

    (leastSquaresAggregator.loss + regVal, new BDV(totalGradientArray))
  }
}
=======
>>>>>>> master
