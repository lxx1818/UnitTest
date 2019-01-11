//package com.test
//
//import com.common.Tools
//import org.apache.spark.ml.feature.FMFeatures
//import org.apache.spark.sql.functions.{col, udf}
//import org.apache.spark.sql.{DataFrame, Dataset, SparkSession}
//
//object FMFeaturesPressureTest {
//
//  val Ciphertext = "\nCiphertext1596876248dfsa "
//
//
//  private val LRFM1 = "LRFM1"
//  private val LRFM2 = "LRFM2"
//  private val LRFM3 = "LRFM3"
//  private val LRFM4 = "LRFM4"
//
//
//  def main(args: Array[String]): Unit = {
////    testNumLabel()
////    testDim1()
////    testDim()
////    testRows()
//    FMFeaturesPrecisionTest.testOnlineRetail()
//  }
//
//  def testRows(): Unit = {
//    print(Ciphertext + "=========测试数据量=========\n")
//    //    val spark = SparkSession.builder().getOrCreate()
//    val spark = Tools.getSparkSession()
//    val path = "/lxxDataSet/FMFeaturesDS/rows/"
//    val names = Array("10", "50", "100", "200", "300", "400", "800", "1600")
////    val names = Array("10")
//    names.foreach { name =>
//      print(Ciphertext + "=========" + name + "k=========\n")
//      val df = spark.read.format("csv")
//        .option("header", "false")
//        .option("inferSchema", true.toString)
//        .load(path + name + "k.csv")
////        .load("/home/lxx/data/FMFeatures集群压测/10k.csv")
//
//      val dataFrame = genLabel(df)
//
//
//
//      val conColName = Tools.getColName(500, 1)
//      val disColName = Tools.getColName(500, 501)
//
//      dataFrame.persist()
//      for (i <- 0 until 5) {
//        val iterationStartTime = System.nanoTime()
//
//        val xgDF = getFMFeatures()
//          .setContinuousFeatureCols(conColName)
//          .setDiscreteFeatureCols(disColName)
//          .setStrategy(LRFM1)
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
//      dataFrame.unpersist()
//    }
//  }
//
//  // 单列增加离散列的标签数
//  def testDim(): Unit = {
//    print(Ciphertext + "=========测试维度=========\n")
//    val spark = Tools.getSparkSession()
//    val path = "/lxxDataSet/FMFeaturesDS/rows/10k.csv"
//    val df = spark.read.format("csv").option("header", "false")
//      .option("inferSchema", true.toString).load(path)
//    val names = Array(1, 5, 10, 20, 40, 80, 160)
//    names.foreach { name =>
//      print(Ciphertext + "=========" + name.toString + "k=========\n")
//      val dataFrame = genDimCol(genLabel(df), name * 1000)
//
//      val conColName = Tools.getColName(500, 1)
//
//      dataFrame.persist()
//      for (i <- 0 until 5) {
//        val iterationStartTime = System.nanoTime()
//
//        val xgDF = getFMFeatures()
//          .setContinuousFeatureCols(conColName)
//          .setDiscreteFeatureCols(Array("discol"))
//          .setStrategy(LRFM1)
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
//      dataFrame.unpersist()
//
//    }
//  }
//
//  // 每个列都是10的标签数
//  def testDim1(): Unit = {
//    print(Ciphertext + "=========测试维度=========\n")
//    val spark = Tools.getSparkSession()
//    val path = "/lxxDataSet/FMFeaturesDS/rows/10k.csv"
//    val df = spark.read.format("csv").option("header", "false")
//      .option("inferSchema", true.toString).load(path)
//    val names = Array(1, 5, 10, 100, 200, 400, 600, 800)
//    val dataFrame = genLabel(df)
//    dataFrame.persist()
//
//    names.foreach { name =>
//      print(Ciphertext + "=========" + name.toString + "k=========\n")
//
//      val conColName = Tools.getColName(100, 901)
//      val disColName = Tools.getColName(name, 1)
//
//      for (i <- 0 until 5) {
//        val iterationStartTime = System.nanoTime()
//
//        val xgDF = getFMFeatures()
//          .setContinuousFeatureCols(conColName)
//          .setDiscreteFeatureCols(disColName)
//          .setStrategy(LRFM1)
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
//
//    dataFrame.unpersist()
//  }
//
//  def testNumLabel(): Unit = {
//    print(Ciphertext + "=========测试numLabel值=========\n")
//    val spark = SparkSession.builder().getOrCreate()
//    val path = "/lxxDataSet/FMFeaturesDS/rows/10k.csv"
//    val df = spark.read.format("csv").option("header", "false")
//      .option("inferSchema", true.toString).load(path)
//    df.persist()
//    val center = Array(2, 4, 8, 16, 32, 64, 96)
//    val conColName = Tools.getColName(500, 1)
//    val disColName = Tools.getColName(500, 501)
//    center.foreach { ce =>
//      print(Ciphertext + s"=========$ce c=========\n")
//      val dataFrame = genLabel(df, ce)
//      for (i <- 0 until 5) {
//        val iterationStartTime = System.nanoTime()
//        val xgDF = getFMFeatures()
//          .setContinuousFeatureCols(conColName)
//          .setDiscreteFeatureCols(disColName)
//          .setStrategy(LRFM1)
//          .fit(dataFrame)
//          .transform(dataFrame)
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
//  private def genLabel(dataset: Dataset[_], n: Int = 2): DataFrame = {
//    val newLabelUDF = udf { label: Double =>
//      label%n
//    }
//    dataset.withColumn("label", newLabelUDF(col("_c0")))
//      .drop("_c0")
//  }
//
//  private def genDimCol(dataset: Dataset[_], num: Int = 2, colName: String = "discol"): DataFrame = {
//    val newDimUDF = udf { label: Double =>
//      (new util.Random).nextInt(num)
//    }
//    dataset.withColumn(colName, newDimUDF(col("_c1")))
//  }
//
//  def getFMFeatures(): FMFeatures ={
//    new FMFeatures()
//      .setSolver("sgda")
//      .setMaxIter(10)
//      .setInitialStd(0.2)
//      .setUseBiasTerm(false)
//      .setUseLinearTerms(false)
//      .setNumFactors(3)
//      .setRegParam0(0)
//      .setRegParam1(0)
//      .setRegParam2(0)
//      .setTol(1e-19)
//      .setThreshold(0.6)
//      .setMinBatchFraction(1.0)
//      .setStepSize(0.15)
//      .setValidationFraction(0.3)
//      .setThresholdLR(0.4)
//      .setMaxIterLR(10)
//      .setTolLR(1e-19)
//      .setRegParamLR(0.0)
//      .setElasticNetParamLR(0.0)
//      .setFitInterceptLR(true)
//      .setStandardizationLR(true)
//      .setAggregationDepthLR(2)
//      .setOutputCol("xgFeatures")
//      .setLabelCol("label")
//  }
//
//}
