//package com.test
//
//import com.common.{OftenUseDataSet, Tools}
//import com.common.OftenUseDataSet._
//import com.common.Tools._
//import org.apache.spark.ml.classification.LogisticRegression
//import org.apache.spark.ml.feature.{FMFeatures, FMFeaturesModel, OneHotEncoderEstimator, VectorAssembler}
//import org.apache.spark.sql.{DataFrame, Dataset}
//
//object FMFeaturesPrecisionTest {
//
//  private val LRFM1 = "LRFM1"
//  private val LRFM2 = "LRFM2"
//  private val LRFM3 = "LRFM3"
//  private val LRFM4 = "LRFM4"
//
//  val Ciphertext = "\nCiphertext1596876248dfsa "
//  val strategies = Array(LRFM1, LRFM2, LRFM3, LRFM4)
//
//  def main(args: Array[String]): Unit = {
//    testOnlineRetail()
//  }
//
//  def testOnlineRetail(): Unit = {
//    var result = ""
//    val spark = getSparkSession()
////    val dataSet = spark.read.format("csv")
////      .option("header", "true")
////      .option("inferSchema", true.toString)
////      .load("/lxxDataSet/FMFeaturesDS/OnlineRetailOK.csv")
//    val dataSet = getOnlineRetail(spark).sample(0.001)
//    val arrData = kFold(dataSet,5)
//
//    arrData.foreach { x =>
//      x._1.persist()
//      x._2.persist()
//    }
//
//    strategies.foreach { strategy =>
//
//      val lrResult = Array.fill(arrData.length)((0.0, 0.0))
//      val fmResult = Array.fill(arrData.length)((0.0, 0.0))
//
//      arrData.indices.foreach { index =>
//        lrResult(index) = lr(arrData(index)._1, arrData(index)._2, true, dataSet)
//        fmResult(index) = fmAndlr(arrData(index)._1, arrData(index)._2, strategy, dataSet)
//      }
//
//      result += strategy + "\n"
//
//      result += lrResult.map(_._1).mkString(",") + "\n"
//      result += lrResult.map(_._2).mkString(",") + "\n"
//      result += fmResult.map(_._1).mkString(",") + "\n"
//      result += fmResult.map(_._2).mkString(",") + "\n"
//
//    }
//
//    arrData.foreach { x =>
//      x._1.unpersist()
//      x._2.unpersist()
//    }
//
//    print(result)
//  }
//
//  def fmAndlr(trainDF: Dataset[_], testDF: Dataset[_], strategy: String, sourceDF: Dataset[_]): (Double, Double) = {
//    trainDF.persist()
//    testDF.persist()
//    val iterationStartTime = System.nanoTime()
//    val model = getFMFeatures().setStrategy(strategy).fit(trainDF)
//    val time = (System.nanoTime() - iterationStartTime) / 1e9
//    val ptrainDF = fff(trainDF, model)
//    val ptestDF = fff(trainDF, model)
//
//    val result = new LogisticRegression()
//      .setThreshold(0.5)
//      .setMaxIter(100)
//      .setTol(1e-6)
//      .setRegParam(0.0)
//      .setElasticNetParam(0.0)
//      .setFitIntercept(true)
//      .setStandardization(true)
//      .setAggregationDepth(2)
//      .setFeaturesCol(featureColName)
//      .setLabelCol(labelColName)
//      .fit(ptrainDF).transform(ptestDF)
//
//    val re = Tools.getAccuracyBC(result)
//
//    trainDF.unpersist()
//    testDF.unpersist()
//    (re, time)
//  }
//
//  // fm 中处理数据
//  def fff(dataset: Dataset[_], fmModel: FMFeaturesModel): DataFrame = {
//    val df = fmModel.transform(dataset).drop(Array(
//      //"InvoiceNo",
//      "CustomerID",
//      "DescriptionIndex",
//      "CountryIndex"): _*)
//    new VectorAssembler()
//      .setInputCols(Array("Date", "Time", "UnitPrice", "xgFeatures"))
//      .setOutputCol(featureColName)
//      .transform(df).drop("Date", "Time", "UnitPrice", "xgFeatures")
//  }
//
//
//
//  def lr(trainDF: Dataset[_], testDF: Dataset[_], isOneHout: Boolean, sourceDF: Dataset[_]): (Double, Double) = {
//
//    trainDF.persist()
//    testDF.persist()
//
//    val (trainDFOH, testDFOH, finalFeaturesName) = if (isOneHout) {
//      val (colName1, traindf) = oneHotFeatures(trainDF, Array(
//        //"InvoiceNo",
//        "CustomerID",
//        "DescriptionIndex",
//        "CountryIndex"), sourceDF)
//      val (colName2, testdf) = oneHotFeatures(testDF, Array(
//        //"InvoiceNo",
//        "CustomerID",
//        "DescriptionIndex",
//        "CountryIndex"), sourceDF)
//      val featuresName = colName1 ++ Array("Date", "Time", "UnitPrice")
//      (traindf, testdf, featuresName)
//    } else {
//      val featuresName = Array("Date", "Time", "UnitPrice",
//        //"InvoiceNo",
//        "CustomerID",
//        "DescriptionIndex",
//        "CountryIndex")
//        (trainDF, testDF, featuresName)
//    }
//
//
//    val trainDFV = new VectorAssembler()
//      .setInputCols(finalFeaturesName)
//      .setOutputCol(featureColName)
//      .transform(trainDFOH).drop(finalFeaturesName:_*)
//    val testDFV = new VectorAssembler()
//      .setInputCols(finalFeaturesName)
//      .setOutputCol(featureColName)
//      .transform(testDFOH).drop(finalFeaturesName:_*)
//
//    val iterationStartTime = System.nanoTime()
//
//    val result = new LogisticRegression()
//      .setThreshold(0.5)
//      .setMaxIter(100)
//      .setTol(1e-6)
//      .setRegParam(0.0)
//      .setElasticNetParam(0.0)
//      .setFitIntercept(true)
//      .setStandardization(true)
//      .setAggregationDepth(2)
//      .setFeaturesCol(OftenUseDataSet.featureColName)
//      .setLabelCol(OftenUseDataSet.labelColName)
//      .fit(trainDFV).transform(testDFV)
//
//    val time = (System.nanoTime() - iterationStartTime) / 1e9
//
//    val re = Tools.getAccuracyBC(result)
//
//    trainDF.unpersist()
//    testDF.unpersist()
//    (re, time)
//  }
//
//  def getFMFeatures(): FMFeatures ={
//    new FMFeatures()
//      .setSolver("sgda")
//      .setMaxIter(20)
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
//      .setMaxIterLR(20)
//      .setTolLR(1e-19)
//      .setRegParamLR(0.0)
//      .setElasticNetParamLR(0.0)
//      .setFitInterceptLR(true)
//      .setStandardizationLR(true)
//      .setAggregationDepthLR(2)
//      .setContinuousFeatureCols(Array("Date", "Time", "UnitPrice"))
//      .setDiscreteFeatureCols(Array(
//        //"InvoiceNo",
//        "CustomerID",
//        "DescriptionIndex",
//        "CountryIndex"))
//      .setOutputCol("xgFeatures")
//      .setLabelCol("label")
//  }
//
//  /**
//    *
//    * @param dataset Training dataset
//    * @param featureCols Discrete feature Cols
//    * @return (New feature column name, Onehot processed data)
//    */
//  def oneHotFeatures(
//      dataset: Dataset[_],
//      featureCols: Array[String],
//      sourceDF: Dataset[_]): (Array[String], DataFrame) = {
//    val oneHotFeatureCols = featureCols.map(_ + "OneHot")
//    val resultDF = new OneHotEncoderEstimator()
//      .setDropLast(false)
//      .setInputCols(featureCols)
//      .setOutputCols(oneHotFeatureCols)
//      .fit(sourceDF)
//      .transform(dataset)
//      .drop(featureCols:_*)
//    (oneHotFeatureCols, resultDF)
//  }
//}
