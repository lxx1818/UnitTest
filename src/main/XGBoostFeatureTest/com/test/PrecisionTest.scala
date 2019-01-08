package com.test

import com.common.{OftenUseDataSet, Tools}
import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.sql.SparkSession

object PrecisionTest {

  def main(args: Array[String]): Unit = {
    print("123")
    testIris()
  }

  def testWine(): Unit ={
    val spark = SparkSession.builder().master("local[*]").appName("CanopyTest").getOrCreate()
    val wineDF = spark.read.format("csv")
      .option("header", "false")
      .option("inferSchema", true.toString)
      .load("/home/lxx/data/wine.data")

    val t = 0
  }

  def testIris(): Unit ={
    val spark = Tools.getSparkSession()

    val irisDF = OftenUseDataSet.getGerman(spark)

    val result = new LogisticRegression()
      .setThreshold(0.5)
      .setMaxIter(30)
      .setTol(1.0e-6)
      .setRegParam(0.0)
      .setElasticNetParam(0.0)
      .setFitIntercept(true)
      .setStandardization(true)
      .setAggregationDepth(2)
      .setFeaturesCol(OftenUseDataSet.featureColName)
      .setLabelCol(OftenUseDataSet.labelColName)
      .fit(irisDF).transform(irisDF)
    val t = 0

  }

}
