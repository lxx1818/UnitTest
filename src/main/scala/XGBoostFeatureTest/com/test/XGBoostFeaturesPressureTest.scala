//package com.test
//
//import com.common.Tools
//import org.apache.spark.ml.feature.XGBoostFeatures
//import org.apache.spark.sql.{DataFrame, Dataset, Row, SparkSession}
//import org.apache.spark.sql.functions.{col, udf}
//
//object XGBoostFeaturesPressureTest {
//
//  val Ciphertext = "\nCiphertext1596876248dfsa "
//
//  def main(args: Array[String]): Unit = {
//    testCores()
//  }
//
//  def testRows(): Unit = {
//    print(Ciphertext + "=========测试数据量=========\n")
////    val spark = SparkSession.builder().getOrCreate()
//    val spark = Tools.getSparkSession()
//    val path = "/canopyData/rows/"
//    val names = Array("10", "50", "100", "200", "300", "400", "800", "1600")
//    names.foreach { name =>
//      print(Ciphertext + "=========" + name + "k=========\n")
//      val df = spark.read.format("libsvm")
//        .load(path + name + "k.libsvm")
////        .load("/home/lxx/data/canopyData/rows/2c.libsvm")
//      val dataFrame = genLabel(df)
//        for (i <- 0 until 5) {
//        val iterationStartTime = System.nanoTime()
//
//          val xgDF = new XGBoostFeatures()
//            .setObjective("binary:logistic")
//            .setNumClasses(2)
//            .setBoosterType("gbtree")
//            .setEta(0.4)
//            .setGamma(0.1)
//            .setMaxDepth(7)
//            .setMinChildWeight(0.0)
//            .setMaxDeltaStep(0.1)
//            .setSubSample(0.9)
//            .setColSampleByTree(0.9)
//            .setColSampleByLevel(0.9)
//            .setLambda(0.9)
//            .setAlpha(0.1)
//            .setTreeMethod("exact")
//            .setSketchEps(0.04)
//            .setScalePosWeight(0.9)
//            .setSampleType("weighted")
//            .setNormalizeType("forest")
//            .setRateDrop(0.1)
//            .setSkipDrop(0.1)
//            .setRound(5)
//            .setOutPutCol("outputVe")
//            .setFeaturesCol("features")
//            .setLabelCol("label")
//            .fit(dataFrame)
//            .transform(dataFrame)
//
//          val iterationTimeInSeconds = (System.nanoTime() - iterationStartTime) / 1e9
//          if (i == 4) {
//            print(Ciphertext + iterationTimeInSeconds + "\n")
//          } else {
//            print(Ciphertext + iterationTimeInSeconds + ",\n")
//          }
//      }
//    }
//  }
//
//  def testDim(): Unit = {
//    print(Ciphertext + "=========测试维度=========\n")
//    val spark = SparkSession.builder().getOrCreate()
//    val path = "/canopyData/dims/"
//    val names = Array("1", "5", "10", "20", "40", "80", "160")
//    names.foreach { name =>
//      print(Ciphertext + "=========" + name + "k=========\n")
//      val df = spark.read.format("libsvm")
//        .load(path + name + "k.libsvm")
//      val dataFrame = genLabel(df)
//      for (i <- 0 until 5) {
//        val iterationStartTime = System.nanoTime()
//
//        val xgDF = new XGBoostFeatures()
//          .setObjective("binary:logistic")
//          .setNumClasses(2)
//          .setBoosterType("gbtree")
//          .setEta(0.4)
//          .setGamma(0.1)
//          .setMaxDepth(7)
//          .setMinChildWeight(0.0)
//          .setMaxDeltaStep(0.1)
//          .setSubSample(0.9)
//          .setColSampleByTree(0.9)
//          .setColSampleByLevel(0.9)
//          .setLambda(0.9)
//          .setAlpha(0.1)
//          .setTreeMethod("exact")
//          .setSketchEps(0.04)
//          .setScalePosWeight(0.9)
//          .setSampleType("weighted")
//          .setNormalizeType("forest")
//          .setRateDrop(0.1)
//          .setSkipDrop(0.1)
//          .setRound(5)
//          .setOutPutCol("outputVe")
//          .setFeaturesCol("features")
//          .setLabelCol("label")
//          .fit(dataFrame)
//          .transform(dataFrame)
//
//        val iterationTimeInSeconds = (System.nanoTime() - iterationStartTime) / 1e9
//        if (i == 4) {
//          print(Ciphertext + iterationTimeInSeconds + "\n")
//        } else {
//          print(Ciphertext + iterationTimeInSeconds + ",\n")
//        }
//      }
//    }
//  }
//
//  def testNumLabel(): Unit = {
//    print(Ciphertext + "=========测试numLabel值=========\n")
//    val spark = SparkSession.builder().getOrCreate()
//    val path = "/canopyData/rows/400k.libsvm"
//    val df = spark.read.format("libsvm").load(path)
//    df.persist()
//    val center = Array(2, 4, 8, 16, 32, 64, 96)
//    center.foreach { ce =>
//      print(Ciphertext + s"=========$ce c=========\n")
//      val dataFrame = genLabel(df, ce)
//      val objective = if (ce > 2) "multi:softprob" else "binary:logistic"
//
//      for (i <- 0 until 5) {
//        val iterationStartTime = System.nanoTime()
//
//        val xgDF = new XGBoostFeatures()
//          .setObjective(objective)
//          .setNumClasses(ce)
//          .setBoosterType("gbtree")
//          .setEta(0.4)
//          .setGamma(0.1)
//          .setMaxDepth(7)
//          .setMinChildWeight(0.0)
//          .setMaxDeltaStep(0.1)
//          .setSubSample(0.9)
//          .setColSampleByTree(0.9)
//          .setColSampleByLevel(0.9)
//          .setLambda(0.9)
//          .setAlpha(0.1)
//          .setTreeMethod("exact")
//          .setSketchEps(0.04)
//          .setScalePosWeight(0.9)
//          .setSampleType("weighted")
//          .setNormalizeType("forest")
//          .setRateDrop(0.1)
//          .setSkipDrop(0.1)
//          .setRound(5)
//          .setOutPutCol("outputVe")
//          .setFeaturesCol("features")
//          .setLabelCol("label")
//          .fit(dataFrame)
//          .transform(dataFrame)
//
//        val iterationTimeInSeconds = (System.nanoTime() - iterationStartTime) / 1e9
//        if (i == 4) {
//          print(Ciphertext + iterationTimeInSeconds + "\n")
//        } else {
//          print(Ciphertext + iterationTimeInSeconds + ",\n")
//        }
//      }
//
//    }
//
//
//    df.unpersist()
//  }
//
//
//  def testCores(): Unit = {
//    print(Ciphertext + "=========测试cores值=========\n")
//    val spark = SparkSession.builder().getOrCreate()
//    val path = "/canopyData/rows/400k.libsvm"
//    val df = spark.read.format("libsvm").load(path)
//    val dataFrame = genLabel(df)
//    for (i <- 0 until 5) {
//      val iterationStartTime = System.nanoTime()
//
//      val xgDF = new XGBoostFeatures()
//        .setObjective("binary:logistic")
//        .setNumClasses(2)
//        .setBoosterType("gbtree")
//        .setEta(0.4)
//        .setGamma(0.1)
//        .setMaxDepth(7)
//        .setMinChildWeight(0.0)
//        .setMaxDeltaStep(0.1)
//        .setSubSample(0.9)
//        .setColSampleByTree(0.9)
//        .setColSampleByLevel(0.9)
//        .setLambda(0.9)
//        .setAlpha(0.1)
//        .setTreeMethod("exact")
//        .setSketchEps(0.04)
//        .setScalePosWeight(0.9)
//        .setSampleType("weighted")
//        .setNormalizeType("forest")
//        .setRateDrop(0.1)
//        .setSkipDrop(0.1)
//        .setRound(5)
//        .setOutPutCol("outputVe")
//        .setFeaturesCol("features")
//        .setLabelCol("label")
//        .fit(dataFrame)
//        .transform(dataFrame)
//
//      val iterationTimeInSeconds = (System.nanoTime() - iterationStartTime) / 1e9
//      if (i == 4) {
//        print(Ciphertext + iterationTimeInSeconds + "\n")
//      } else {
//        print(Ciphertext + iterationTimeInSeconds + ",\n")
//      }
//    }
//  }
//
//
//  private def genLabel(dataset: Dataset[_], n: Int = 2): DataFrame = {
//    val newLabelUDF = udf { label: Double =>
//      label%n
//    }
//    dataset.withColumn("label", newLabelUDF(col("label")))
//  }
//
//}
