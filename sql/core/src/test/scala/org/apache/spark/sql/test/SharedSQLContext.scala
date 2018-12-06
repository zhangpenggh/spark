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

package org.apache.spark.sql.test

<<<<<<< HEAD
import scala.concurrent.duration._

import org.scalatest.BeforeAndAfterEach
import org.scalatest.concurrent.Eventually

import org.apache.spark.{DebugFilesystem, SparkConf}
import org.apache.spark.sql.{SparkSession, SQLContext}

/**
 * Helper trait for SQL test suites where all tests share a single [[TestSparkSession]].
 */
trait SharedSQLContext extends SQLTestUtils with BeforeAndAfterEach with Eventually {

  protected def sparkConf = {
    new SparkConf().set("spark.hadoop.fs.file.impl", classOf[DebugFilesystem].getName)
  }
=======
trait SharedSQLContext extends SQLTestUtils with SharedSparkSession {
>>>>>>> master

  /**
   * Suites extending [[SharedSQLContext]] are sharing resources (eg. SparkSession) in their tests.
   * That trait initializes the spark session in its [[beforeAll()]] implementation before the
   * automatic thread snapshot is performed, so the audit code could fail to report threads leaked
   * by that shared session.
   *
   * The behavior is overridden here to take the snapshot before the spark session is initialized.
   */
  override protected val enableAutoThreadAudit = false

<<<<<<< HEAD
  protected def createSparkSession: TestSparkSession = {
    new TestSparkSession(sparkConf)
  }

  /**
   * Initialize the [[TestSparkSession]].
   */
=======
>>>>>>> master
  protected override def beforeAll(): Unit = {
    doThreadPreAudit()
    super.beforeAll()
  }

  protected override def afterAll(): Unit = {
    super.afterAll()
<<<<<<< HEAD
    if (_spark != null) {
      _spark.sessionState.catalog.reset()
      _spark.stop()
      _spark = null
    }
  }

  protected override def beforeEach(): Unit = {
    super.beforeEach()
    DebugFilesystem.clearOpenStreams()
  }

  protected override def afterEach(): Unit = {
    super.afterEach()
    // files can be closed from other threads, so wait a bit
    // normally this doesn't take more than 1s
    eventually(timeout(10.seconds)) {
      DebugFilesystem.assertNoOpenStreams()
    }
=======
    doThreadPostAudit()
>>>>>>> master
  }
}
