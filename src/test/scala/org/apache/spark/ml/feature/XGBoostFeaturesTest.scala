package org.apache.spark.ml.extension.feature

import ml.dmlc.xgboost4j.scala.spark.XGBoostEstimator
import org.apache.spark.ml.GeneralFunSuite
import org.apache.spark.ml.feature.{VectorAssembler, XGBoostFeatures}
import org.apache.spark.ml.linalg.Vector
import org.apache.spark.ml.param._
import org.apache.spark.sql.{DataFrame, Row, SparkSession}


class XGBoostFeaturesTest extends GeneralFunSuite {

  var inputDF: DataFrame = _

  override def beforeAll(): Unit = {
    super.beforeAll()
    val df = XGBoostFeaturesTest.generateTestData(spark)
    val vectorAssembler = new VectorAssembler()
      .setInputCols(Array("col1", "col2", "col3"))
      .setOutputCol("featuresVe")
    inputDF = vectorAssembler.transform(df).select("featuresVe", "labelCol")
  }

  test("set params") {
    val xGBoostFeatures = new XGBoostFeatures()
      .setObjective("binary:logistic")
      .setNumClasses(2)
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
      .setRound(20)
      .setOutPutCol("outputVe")
      .setFeaturesCol("featuresVe")
      .setLabelCol("labelCol")
    val xgbFeaturesModel = xGBoostFeatures.fit(inputDF)
    val xgbModel = xgbFeaturesModel.xgbModel.parent
    def $[T](param: Param[T]): T = xgbModel.getOrDefault(param)
    val estimator = new XGBoostEstimator(Map[String, Any]())

    assert($(xgbModel.getParam(estimator.objective.name)) === "binary:logistic")
    assert($(xgbModel.getParam(estimator.numClasses.name)) === 2)
    assert($(xgbModel.getParam(estimator.boosterType.name)) === "gbtree")
    assert($(xgbModel.getParam(estimator.eta.name)) === 0.4)
    assert($(xgbModel.getParam(estimator.gamma.name)) === 0.1)
    assert($(xgbModel.getParam(estimator.maxDepth.name)) === 7)
    assert($(xgbModel.getParam(estimator.minChildWeight.name)) === 0.0)
    assert($(xgbModel.getParam(estimator.maxDeltaStep.name)) === 0.1)
    assert($(xgbModel.getParam(estimator.subSample.name)) === 0.9)
    assert($(xgbModel.getParam(estimator.colSampleByTree.name)) === 0.9)
    assert($(xgbModel.getParam(estimator.colSampleByLevel.name)) === 0.9)
    assert($(xgbModel.getParam(estimator.lambda.name)) === 0.9)
    assert($(xgbModel.getParam(estimator.alpha.name)) === 0.1)
    assert($(xgbModel.getParam(estimator.treeMethod.name)) === "exact")
    assert($(xgbModel.getParam(estimator.sketchEps.name)) === 0.04)
    assert($(xgbModel.getParam(estimator.scalePosWeight.name)) === 0.9)
    assert($(xgbModel.getParam(estimator.sampleType.name)) === "weighted")
    assert($(xgbModel.getParam(estimator.normalizeType.name)) === "forest")
    assert($(xgbModel.getParam(estimator.rateDrop.name)) === 0.1)
    assert($(xgbModel.getParam(estimator.skipDrop.name)) === 0.1)
    assert($(xgbModel.getParam(estimator.round.name)) === 20)
    assert(xgbFeaturesModel.xgbModel.getFeaturesCol === "featuresVe")
    assert(xgbFeaturesModel.xgbModel.getLabelCol === "labelCol")
    assert(xgbFeaturesModel.getOutputCol === "outputVe")
  }

  test("Test illegal input") {
    // No features Col
    var t = intercept[Exception] {
      new XGBoostFeatures()
        .setLabelCol("labelCol")
        .fit(inputDF)
    }
    assert(t.getMessage === "Field \"features\" does not exist.")

    // Wrong features Col
    t = intercept[Exception] {
      new XGBoostFeatures()
        .setFeaturesCol("fff")
        .setLabelCol("labelCol")
        .fit(inputDF)
    }
    assert(t.getMessage === "Field \"fff\" does not exist.")

    // No label Col
    t = intercept[Exception] {
      new XGBoostFeatures()
        .setFeaturesCol("featuresVe")
        .fit(inputDF)
    }
    assert(t.getMessage === "Field \"label\" does not exist.")

    // Wrong label Col
    t = intercept[Exception] {
      new XGBoostFeatures()
        .setFeaturesCol("featuresVe")
        .setLabelCol("aaa")
        .fit(inputDF)
    }
    assert(t.getMessage === "Field \"aaa\" does not exist.")
  }

  test("Test for schema && outcome") {
    val xgbFeatureModel = new XGBoostFeatures()
      .setFeaturesCol("featuresVe")
      .setLabelCol("labelCol")
      .setOutPutCol("XFOutput")
      .fit(inputDF)
    val resultSchema = xgbFeatureModel.transformSchema(inputDF.schema)
    val outCome = xgbFeatureModel.transform(inputDF)

    // validate schema
    val outputColNames = resultSchema.map(_.name)
    assert(outputColNames === Array("featuresVe", "labelCol", "XFOutput"))

    val numValidTrees = xgbFeatureModel.leafIndexMaps.count(_.size > 1)
    val countArr = outCome.select("XFOutput").collect().map {
      case Row(v: Vector) => v.toArray.sum.toInt
    }
    assert(countArr.forall(_ == numValidTrees))
  }

  test("Test for outcome of XGBoostFeatures") {
    val xgbFeatureModel = new XGBoostFeatures()
      .setRound(10)
      .setMinChildWeight(0)
      .setMaxDepth(5)
      .setRound(3)
      .setFeaturesCol("featuresVe")
      .setLabelCol("labelCol")
      .setOutPutCol("XFOutput")
      .setSubSample(1.0)
      .setColSampleByLevel(1.0)
      .fit(inputDF)
    val outCome = xgbFeatureModel.transform(inputDF)
    val result = outCome.select("XFOutput").collect().map {
      case Row(v: Vector) => v.toArray
    }


    // predict model is 3 + 3 + 3
    assert(result(0) === Array(0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0))
    assert(result(1) === Array(0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0))
    assert(result(2) === Array(1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0))
    assert(result(3) === Array(0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0))
    assert(result(4) === Array(0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0))
    assert(result(5) === Array(0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0))
    assert(result(6) === Array(1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0))
    assert(result(7) === Array(1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0))
  }
}

object XGBoostFeaturesTest {
  def generateTestData(spark: SparkSession): DataFrame = {
    import spark.implicits._
    val testDF = Seq(
      (1.0, 0.0, 1.0, 1.0),
      (1.0, 0.0, 1.0, 1.0),
      (0.0, 1.0, 0.0, 0.0),
      (0.0, 0.0, 1.0, 1.0),
      (1.0, 0.0, 1.0, 0.0),
      (0.0, 0.0, 1.0, 0.0),
      (0.0, 1.0, 0.0, 0.0),
      (1.0, 0.0, 0.0, 0.0)
    ).toDF("col1", "col2", "col3", "labelCol")
    testDF
  }
}
