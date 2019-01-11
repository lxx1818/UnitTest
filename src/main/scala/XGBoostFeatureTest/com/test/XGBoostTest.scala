//package com.test
//
//import com.common.{OftenUseDataSet, Tools}
//import ml.dmlc.xgboost4j.scala.spark.{ImproveXGBoostClassificationModel, XGBoostClassificationModel, XGBoostEstimator}
//import org.apache.spark.sql.DataFrame
//
//object XGBoostTest {
//
//  def main(args: Array[String]): Unit = {
//    var s = ""
//
////    val arr = Array(0.0, 1.0, 2.0, 3.0, 4.0, 5.0)
////    val arr = Array(1e-10, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1 - 1e-10)
////    val arr = Array(0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.99)
////    val arr = getArr(0.9, 1)
////    val arr = Array("auto", "exact", "approx")
////    val arr = Array("gbtree", "gblinear", "dart")
////    val arr = Array("uniform", "weighted")
////    val arr = Array("tree", "forest")
////    val arr = Array(1, 2, 3, 4, 5, 6, 7, 8, 9, 10)
//    val arr = Array(100, 200, 300, 400, 500, 600, 700, 800, 900, 1000).map(_/10)
//
//    val result = Array.fill(arr.length)(0.0)
//
//    arr.indices.foreach { index =>
//      result(index) = setParamHaberman(arr(index))
//    }
//    s += arr.zip(result).mkString("\n")
////    s += arr.mkString(", ") + "\n"
////    s += result.mkString(", ")
//
//    Tools.getSparkSession().close()
//    Thread.sleep(1000)
//    print("\n\n\n" + s + "\n\n\n")
//
//  }
//
//  def setParamHaberman(value: Int): Double = {
//
//    var result = ""
//    val spark = Tools.getSparkSession()
//    val habermanDF = OftenUseDataSet.getGerman(spark)
//    habermanDF.persist()
//
//    val df = Tools.kFold(habermanDF)
//
//    val estimator = new XGBoostEstimator(Map[String, Any]())
//    val paramMap = Map(
//      estimator.round.name -> 100,
//      estimator.objective.name -> "binary:logistic",
//      estimator.numClasses.name -> 2,
//      estimator.boosterType.name -> "gbtree",
//      estimator.eta.name -> 0.52,
//      estimator.gamma.name -> 0.0,
//      estimator.maxDepth.name -> 3,
//      estimator.minChildWeight.name -> 0.5,
//      estimator.maxDeltaStep.name -> 0.1,
//      estimator.subSample.name -> 0.965,
//      estimator.colSampleByTree.name -> 0.98,
//      estimator.colSampleByLevel.name -> 0.5,
//      estimator.lambda.name -> 1.0,
//      estimator.alpha.name -> 0.95,
//      estimator.treeMethod.name -> "auto",
//      estimator.sketchEps.name -> 0.18,
//      estimator.scalePosWeight.name -> 0.968,
//      estimator.skipDrop.name -> 0.0,
//      estimator.rateDrop.name -> 0.0,
//      estimator.sampleType.name -> "weighted",
//      estimator.normalizeType.name -> "forest"
//
//      )
//    val xgb = new XGBoostEstimator(paramMap)
//      .setFeaturesCol(OftenUseDataSet.featureColName)
//      .setLabelCol(OftenUseDataSet.labelColName)
//
//    dealWithResult(df, xgb)
//
//  }
//
//  def dealWithResult(dataset: Array[(DataFrame, DataFrame)], xgboost: XGBoostEstimator): Double = {
//    var sum = 0.0
//
//    dataset.foreach { case (trainDF, testDF) =>
//      val model = xgboost.fit(trainDF).asInstanceOf[XGBoostClassificationModel]
//        val newModel = new ImproveXGBoostClassificationModel(model).copyValues(model)
//      sum += Tools.getAccuracyBC(newModel.transform(testDF))
//    }
//
//    sum / dataset.length
//  }
//
//
//  def getArr(min: Double, count: Int = 20): Array[Double] = {
//    var result = Array.fill(count)(0.0)
//    result.indices.foreach { index =>
//      result(index) = index * 0.005 + min
//    }
//    result
//  }
//
//}
