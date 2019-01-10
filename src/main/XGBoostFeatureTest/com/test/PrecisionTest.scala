package com.test

import com.common.OftenUseDataSet.featureColName
import com.common.{OftenUseDataSet, Tools}
import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.ml.feature.{VectorAssembler, XGBoostFeatures}
import org.apache.spark.sql.{DataFrame, Dataset, SparkSession}

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

  def testIris(): Unit = {

    val spark = Tools.getSparkSession()

    val result1 = Array.fill(5)(0.0)
    val result2 = Array.fill(5)(0.0)

    val dataset = OftenUseDataSet.getGerman(spark)
    for (i <- 0 until 5) {
      result1(i) = lr(dataset)
      result2(i) = xgAndLR(dataset)
    }
    val x = result1.sum / 5.0
    val y = result2.sum / 5.0

    val p = (y - x) / x * 100

    print("OK")

  }

  def xgAndLR(dataset: Dataset[_]): Double = {

    val numClass = Tools.getNumClasses(dataset)
    val objective = if (numClass > 2) "multi:softprob" else "binary:logistic"

    val xgDF = new XGBoostFeatures()
      .setObjective(objective)
      .setNumClasses(numClass)
      .setBoosterType("gbtree")
      .setEta(0.4)
      .setGamma(0.1)
      .setMaxDepth(7)
      .setMinChildWeight(0.0)
      .setMaxDeltaStep(0.1)
      .setSubSample(0.9)
      .setColSampleByTree(0.9)
      .setColSampleByLevel(0.9)
      .setLambda(0.9)
      .setAlpha(0.1)
      .setTreeMethod("exact")
      .setSketchEps(0.04)
      .setScalePosWeight(0.9)
      .setSampleType("weighted")
      .setNormalizeType("forest")
      .setRateDrop(0.1)
      .setSkipDrop(0.1)
      .setRound(5)
      .setOutPutCol("outputVe")
      .setFeaturesCol(OftenUseDataSet.featureColName)
      .setLabelCol(OftenUseDataSet.labelColName)
      .fit(dataset)
      .transform(dataset)

    val resultDF = new VectorAssembler()
      .setInputCols(Array(OftenUseDataSet.featureColName, "outputVe"))
      .setOutputCol("xgfeature")
      .transform(xgDF)

    val result = new LogisticRegression()
      .setThreshold(0.5)
      .setMaxIter(100)
      .setTol(1.0e-6)
      .setRegParam(0.0)
      .setElasticNetParam(0.0)
      .setFitIntercept(true)
      .setStandardization(true)
      .setAggregationDepth(2)
      .setFeaturesCol("xgfeature")
      .setLabelCol(OftenUseDataSet.labelColName)
      .fit(resultDF).transform(resultDF)

    Tools.genAccuracyMC(result)
  }

  def lr(dataset: Dataset[_]): Double = {

    val result = new LogisticRegression()
      .setThreshold(0.5)
      .setMaxIter(100)
      .setTol(1e-6)
      .setRegParam(0.0)
      .setElasticNetParam(0.0)
      .setFitIntercept(true)
      .setStandardization(true)
      .setAggregationDepth(2)
      .setFeaturesCol(OftenUseDataSet.featureColName)
      .setLabelCol(OftenUseDataSet.labelColName)
      .fit(dataset).transform(dataset)

    Tools.genAccuracyMC(result)
  }

}
