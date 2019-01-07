package org.apache.spark.ml

import org.apache.spark.SparkContext
import org.apache.spark.sql.SparkSession
import org.scalatest.{BeforeAndAfterAll, FunSuite}

trait GeneralFunSuite extends FunSuite with BeforeAndAfterAll{
  @transient var spark: SparkSession = _
  @transient var sc: SparkContext = _
  override def beforeAll() {
    super.beforeAll()

    spark = SparkSession.builder()
      .master("local[*]")
      .appName("SparkExtensionUnitTest")
      .config("spark.executor.memory", "2g")
      .getOrCreate()
    sc = spark.sparkContext

  }

  override def afterAll() {
    try {
      SparkSession.clearActiveSession()
      if (spark != null) {
        spark.stop()
      }
      spark = null
    } finally {
      super.afterAll()
    }
  }

}
