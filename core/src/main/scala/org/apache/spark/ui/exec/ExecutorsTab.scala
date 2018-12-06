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

package org.apache.spark.ui.exec

import javax.servlet.http.HttpServletRequest

<<<<<<< HEAD
import org.apache.spark.{Resubmitted, SparkConf, SparkContext}
import org.apache.spark.annotation.DeveloperApi
import org.apache.spark.scheduler._
import org.apache.spark.storage.{StorageStatus, StorageStatusListener}
import org.apache.spark.ui.{SparkUI, SparkUITab}

private[ui] class ExecutorsTab(parent: SparkUI) extends SparkUITab(parent, "executors") {
  val listener = parent.executorsListener
  val sc = parent.sc
  val threadDumpEnabled =
    sc.isDefined && parent.conf.getBoolean("spark.ui.threadDumpsEnabled", true)

  attachPage(new ExecutorsPage(this, threadDumpEnabled))
  if (threadDumpEnabled) {
    attachPage(new ExecutorThreadDumpPage(this))
  }
}

private[ui] case class ExecutorTaskSummary(
    var executorId: String,
    var totalCores: Int = 0,
    var tasksMax: Int = 0,
    var tasksActive: Int = 0,
    var tasksFailed: Int = 0,
    var tasksComplete: Int = 0,
    var duration: Long = 0L,
    var jvmGCTime: Long = 0L,
    var inputBytes: Long = 0L,
    var inputRecords: Long = 0L,
    var outputBytes: Long = 0L,
    var outputRecords: Long = 0L,
    var shuffleRead: Long = 0L,
    var shuffleWrite: Long = 0L,
    var executorLogs: Map[String, String] = Map.empty,
    var isAlive: Boolean = true,
    var isBlacklisted: Boolean = false
)

/**
 * :: DeveloperApi ::
 * A SparkListener that prepares information to be displayed on the ExecutorsTab
 */
@DeveloperApi
@deprecated("This class will be removed in a future release.", "2.2.0")
class ExecutorsListener(storageStatusListener: StorageStatusListener, conf: SparkConf)
    extends SparkListener {
  val executorToTaskSummary = LinkedHashMap[String, ExecutorTaskSummary]()
  var executorEvents = new ListBuffer[SparkListenerEvent]()

  private val maxTimelineExecutors = conf.getInt("spark.ui.timeline.executors.maximum", 1000)
  private val retainedDeadExecutors = conf.getInt("spark.ui.retainedDeadExecutors", 100)

  def activeStorageStatusList: Seq[StorageStatus] = storageStatusListener.storageStatusList
=======
import scala.xml.Node

import org.apache.spark.ui.{SparkUI, SparkUITab, UIUtils, WebUIPage}
>>>>>>> master

private[ui] class ExecutorsTab(parent: SparkUI) extends SparkUITab(parent, "executors") {

<<<<<<< HEAD
  override def onTaskEnd(
      taskEnd: SparkListenerTaskEnd): Unit = synchronized {
    val info = taskEnd.taskInfo
    if (info != null) {
      val eid = info.executorId
      val taskSummary = executorToTaskSummary.getOrElseUpdate(eid, ExecutorTaskSummary(eid))
      // Note: For resubmitted tasks, we continue to use the metrics that belong to the
      // first attempt of this task. This may not be 100% accurate because the first attempt
      // could have failed half-way through. The correct fix would be to keep track of the
      // metrics added by each attempt, but this is much more complicated.
      if (taskEnd.reason == Resubmitted) {
        return
      }
      if (info.successful) {
        taskSummary.tasksComplete += 1
      } else {
        taskSummary.tasksFailed += 1
      }
      if (taskSummary.tasksActive >= 1) {
        taskSummary.tasksActive -= 1
      }
      taskSummary.duration += info.duration
=======
  init()
>>>>>>> master

  private def init(): Unit = {
    val threadDumpEnabled =
      parent.sc.isDefined && parent.conf.getBoolean("spark.ui.threadDumpsEnabled", true)

    attachPage(new ExecutorsPage(this, threadDumpEnabled))
    if (threadDumpEnabled) {
      attachPage(new ExecutorThreadDumpPage(this, parent.sc))
    }
  }

}

private[ui] class ExecutorsPage(
    parent: SparkUITab,
    threadDumpEnabled: Boolean)
  extends WebUIPage("") {

  def render(request: HttpServletRequest): Seq[Node] = {
    val content =
      <div>
        {
          <div id="active-executors" class="row-fluid"></div> ++
          <script src={UIUtils.prependBaseUri(request, "/static/utils.js")}></script> ++
          <script src={UIUtils.prependBaseUri(request, "/static/executorspage.js")}></script> ++
          <script>setThreadDumpEnabled({threadDumpEnabled})</script>
        }
      </div>

    UIUtils.headerSparkPage(request, "Executors", content, parent, useDataTables = true)
  }
}
