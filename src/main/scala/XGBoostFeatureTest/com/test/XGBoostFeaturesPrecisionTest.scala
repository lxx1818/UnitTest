//package com.test
//
//import com.common.{OftenUseDataSet, Tools}
//
//import org.apache.spark.ml.classification.LogisticRegression
//import org.apache.spark.ml.feature.{VectorAssembler, XGBoostFeatures}
//import org.apache.spark.sql.{DataFrame, Dataset}
//
//object XGBoostFeaturesPrecisionTest {
//
//  var dropOrange = false
//
//  def main(args: Array[String]): Unit = {
//    print(test())
//  }
//
//  def test(): String = {
//    var result = ""
//
//    val spark = Tools.getSparkSession()
//
////    val irisDF = OftenUseDataSet.getIrisDataSet(spark).filter(row => row.getDouble(1) < 2)
////    irisDF.persist()
////    result += dealWithResult("iris", Tools.kFold(irisDF))
////    irisDF.unpersist()
////
////    val wineDF = OftenUseDataSet.getWineDataSet(spark).filter(row => row.getDouble(1) < 2)
////    wineDF.persist()
////    result += dealWithResult("wine", Tools.kFold(wineDF))
////    wineDF.unpersist()
//
////    val habermanDF = OftenUseDataSet.getHaberman(spark)
////    habermanDF.persist()
////    result += dealWithResult("haberman", Tools.kFold(habermanDF))
////    habermanDF.unpersist()
//
//    val germanDF = OftenUseDataSet.getGerman(spark)
//    germanDF.persist()
//    result += dealWithResult("german", Tools.kFold(germanDF))
//    germanDF.unpersist()
//
//    Thread.sleep(2000)
//    result
//  }
//
//  def xgAndLR(trainDF: Dataset[_], testDF: Dataset[_], numClass: Int, datasetName: String): Double = {
//    val objective = if (numClass > 2) "multi:softprob" else "binary:logistic"
//
//    val model = datasetName match {
//      case "haberman" => {
//        new XGBoostFeatures()
//          .setObjective(objective)
//          .setNumClasses(numClass)
//          .setBoosterType("gbtree")
//          .setEta(0.2)
//          .setGamma(0.0)
//          .setMaxDepth(3)
//          .setMinChildWeight(0.5)
//          .setMaxDeltaStep(0.1)
//          .setSubSample(0.1)
//          .setColSampleByTree(1.0)
//          .setColSampleByLevel(0.5)
//          .setLambda(1.0)
//          .setAlpha(0.92)
//          .setTreeMethod("auto")
//          .setSketchEps(0.18)
//          .setScalePosWeight(0.968)
//          .setSampleType("uniform")
//          .setNormalizeType("forest")
//          .setRateDrop(0.0)
//          .setSkipDrop(0.0)
//          .setRound(100)
//          .setOutPutCol("outputVe")
//          .setFeaturesCol(OftenUseDataSet.featureColName)
//          .setLabelCol(OftenUseDataSet.labelColName)
//          .fit(trainDF)
//      }
//      case "german" => {
//        new XGBoostFeatures()
//          .setObjective(objective)
//          .setNumClasses(numClass)
//          .setBoosterType("gbtree")
//          .setEta(0.52)
//          .setGamma(0.0)
//          .setMaxDepth(3)
//          .setMinChildWeight(0.5)
//          .setMaxDeltaStep(0.1)
//          .setSubSample(0.965)
//          .setColSampleByTree(0.98)
//          .setColSampleByLevel(0.5)
//          .setLambda(1.0)
//          .setAlpha(0.95)
//          .setTreeMethod("auto")
//          .setSketchEps(0.18)
//          .setScalePosWeight(0.968)
//          .setSampleType("weighted")
//          .setNormalizeType("forest")
//          .setRateDrop(0.0)
//          .setSkipDrop(0.0)
//          .setRound(100)
//          .setOutPutCol("outputVe")
//          .setFeaturesCol(OftenUseDataSet.featureColName)
//          .setLabelCol(OftenUseDataSet.labelColName)
//          .fit(trainDF)
//      }
//      case _ => {
//        new XGBoostFeatures()
//          .setObjective(objective)
//          .setNumClasses(numClass)
//          .setBoosterType("gbtree")
//          .setEta(0.4)
//          .setGamma(0.1)
//          .setMaxDepth(7)
//          .setMinChildWeight(1.0)
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
//          .setRound(100)
//          .setOutPutCol("outputVe")
//          .setFeaturesCol(OftenUseDataSet.featureColName)
//          .setLabelCol(OftenUseDataSet.labelColName)
//          .fit(trainDF)
//      }
//    }
//
//    if (dropOrange) {
//      lr(model.transform(trainDF), model.transform(testDF), featureColName = "outputVe")
//    } else {
//
//      val vectorModel = new VectorAssembler()
//        .setInputCols(Array(OftenUseDataSet.featureColName, "outputVe"))
//        .setOutputCol("newFeatures")
//
//      val newTrainDF = vectorModel.transform(model.transform(trainDF))
//      val newTestDF = vectorModel.transform(model.transform(testDF))
//
//      lr(newTrainDF, newTestDF, featureColName = "newFeatures")
//    }
//
//
//  }
//
//  def lr(
//      trainDF: Dataset[_],
//      testDF: Dataset[_],
//      featureColName: String = OftenUseDataSet.featureColName,
//      labelColName: String = OftenUseDataSet.labelColName): Double = {
//
//    val result = new LogisticRegression()
//      .setThreshold(0.7)
//      .setMaxIter(200)
//      .setTol(1e-15)
//      .setRegParam(0.0)
//      .setElasticNetParam(0.0)
//      .setFitIntercept(true)
//      .setStandardization(true)
//      .setAggregationDepth(2)
//      .setFeaturesCol(featureColName)
//      .setLabelCol(labelColName)
//      .fit(trainDF).transform(testDF)
//
//    Tools.getAccuracyBC(result)
//  }
//
//  def dealWithResult(
//      dataName: String,
//      datasetK: Array[(DataFrame, DataFrame)]): String = {
//    var result = "测试" + dataName + "数据集\n"
//    val kLength = datasetK.length
//    val lrResult = Array.fill(kLength)(0.0)
//    val newResult = Array.fill(kLength)(0.0)
//    val numClass = Tools.getNumClasses(datasetK(0)._1)
//
//    datasetK.zipWithIndex.foreach { case ((trainDF, testDF), index) =>
//      lrResult(index) = lr(trainDF, testDF)
//      newResult(index) = xgAndLR(trainDF, testDF, numClass, dataName)
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
//}
