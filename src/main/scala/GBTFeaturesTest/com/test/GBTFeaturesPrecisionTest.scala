//package com.test
//
//import com.common.{OftenUseDataSet, Tools}
//import org.apache.spark.ml.classification.LogisticRegression
//import org.apache.spark.ml.feature.{GBTFeatures, GBTFeaturesModel, VectorAssembler}
//import org.apache.spark.sql.{DataFrame, Dataset}
//
//object GBTFeaturesPrecisionTest {
//
//  val gbtFeatureColName = "gbtFeature"
//  var dropOrange = false
//
//  def main(args: Array[String]): Unit = {
////    val result = test()
//    val result = XGBoostFeaturesPrecisionTest.test()
//    Tools.getSparkSession().close()
//    Thread.sleep(10000)
//    print(result)
//  }
//
//  def test(): String = {
//    var result = ""
//
//    val spark = Tools.getSparkSession()
//
//    val irisDF = OftenUseDataSet.getIrisDataSet(spark).filter(row => row.getDouble(1) < 2)
//    irisDF.persist()
//    result += dealWithResult("iris", Tools.kFold(irisDF))
//    irisDF.unpersist()
//
//    val wineDF = OftenUseDataSet.getWineDataSet(spark).filter(row => row.getDouble(1) < 2)
//    wineDF.persist()
//    result += dealWithResult("wine", Tools.kFold(wineDF))
//    wineDF.unpersist()
//
////    val habermanDF = OftenUseDataSet.getHaberman(spark)
////    habermanDF.persist()
////    result += dealWithResult("haberman", Tools.kFold(habermanDF))
////    habermanDF.unpersist()
////
////    val germanDF = OftenUseDataSet.getGerman(spark)
////    germanDF.persist()
////    result += dealWithResult("german", Tools.kFold(germanDF))
////    germanDF.unpersist()
//
//    Thread.sleep(2000)
//
//    result
//  }
//
//  def lr(
//      trainDF: Dataset[_],
//      testDF: Dataset[_],
//      featureColName: String = OftenUseDataSet.featureColName,
//      labelColName: String = OftenUseDataSet.labelColName): Double = {
//
//    val result = new LogisticRegression()
//      .setThreshold(0.5)
//      .setMaxIter(50)
//      .setTol(1e-5)
//      .setRegParam(0.0)
//      .setElasticNetParam(0.0)
//      .setFitIntercept(true)
//      .setStandardization(false)
//      .setAggregationDepth(2)
//      .setFeaturesCol(featureColName)
//      .setLabelCol(labelColName)
//      .fit(trainDF).transform(testDF)
//
//    Tools.getAccuracyBC(result)
//  }
//
//  def gbdtAndLR(trainDF: Dataset[_], testDF: Dataset[_]): Double = {
//
//    val model = getGBTFeaturesModel(trainDF)
//    if (dropOrange) {
//      lr(model.transform(trainDF), model.transform(testDF), featureColName = "gbtFeatureColName")
//    } else {
//      val vectorModel = new VectorAssembler()
//        .setInputCols(Array( OftenUseDataSet.featureColName, gbtFeatureColName))
//        .setOutputCol("newFeatures")
//      val newTrainDF = vectorModel.transform(model.transform(trainDF))
//      val newTestDF = vectorModel.transform(model.transform(testDF))
//      lr(newTrainDF, newTestDF, featureColName = "newFeatures")
//    }
//  }
//
//  def getGBTFeaturesModel(trainDF: Dataset[_]): GBTFeaturesModel = {
//    new GBTFeatures()
//      .setTaskType("classification")
//      .setMaxDepth(6)
//      .setSeed(123L)
//      .setMaxIter(100)
//      .setStepSize(0.5)
//      .setFeatureSubsetStrategy("all")
//      .setFeaturesCol(OftenUseDataSet.featureColName)
//      .setLabelCol(OftenUseDataSet.labelColName)
//      .setOutPutCol(gbtFeatureColName)
//      .fit(trainDF)
//  }
//
//  def dealWithResult(
//      dataName: String,
//      datasetK: Array[(DataFrame, DataFrame)]): String = {
//    var result = "测试" + dataName + "数据集\n"
//    val kLength = datasetK.length
//    val lrResult = Array.fill(kLength)(0.0)
//    val newResult = Array.fill(kLength)(0.0)
//
//    datasetK.zipWithIndex.foreach { case ((trainDF, testDF), index) =>
//      lrResult(index) = lr(trainDF, testDF)
//      newResult(index) = gbdtAndLR(trainDF, testDF)
//    }
//
//    val lrAverage = lrResult.sum / kLength
//    val newAverage = newResult.sum / kLength
//
//    result += "LR: " + lrResult.mkString(",") + "\n"
//    result += "NE: " + newResult.mkString(",") + "\n"
//    result += "平均\nLR: " + lrAverage + "\nNE: " + newAverage + "\n"
//    result += "提升(NE - LR) / LR: " + (100 * (newAverage - lrAverage) / lrAverage).formatted("%.2f") + "%\n\n\n"
//    result
//  }
//
//}
