package org.apache.spark.ml.feature

import org.apache.spark.ml.GeneralFunSuite
import org.apache.spark.ml.linalg.Vector
import org.apache.spark.sql.{DataFrame, Row, SparkSession}

class FMFeaturesTest extends GeneralFunSuite {

  var inputDF: DataFrame = _

  override def beforeAll() {
    super.beforeAll()
    inputDF = FMFeaturesTest.generateTestData(spark)
  }

  test("set params") {
    val fMFeatures = new FMFeatures()
      .setSolver("sgda")
      .setMaxIter(50)
      .setInitialStd(0.2)
      .setUseBiasTerm(false)
      .setUseLinearTerms(false)
      .setNumFactors(3)
      .setRegParam0(0)
      .setRegParam1(0)
      .setRegParam2(0)
      .setTol(1e-7)
      .setThreshold(0.6)
      .setMinBatchFraction(0.5)
      .setStepSize(1.5)
      .setValidationFraction(0.3)

      .setThresholdLR(0.4)
      .setMaxIterLR(20)
      .setTolLR(1e-9)
      .setRegParamLR(0.0)
      .setElasticNetParamLR(0.0)
      .setFitInterceptLR(true)
      .setStandardizationLR(true)
      .setAggregationDepthLR(2)
      .setWeightColLR("col4")

      .setStrategy(FMFeatures.LRFM2)
      .setNumBuckets(6)
      .setContinuousFeatureCols(Array("col1", "col2"))
      .setDiscreteFeatureCols(Array("col3"))
      .setOutputCol("newFeature")
      .setLabelCol("label")
    val model = fMFeatures.fit(inputDF)
  }

  test("Test illegal input") {
    // No discreteFeatureCols
    var t = intercept[Exception] {
      new FMFeatures()
        .setStrategy(FMFeatures.LRFM2)
        .setContinuousFeatureCols(Array("col1", "col2"))
        .fit(inputDF)
    }
    assert(t.getMessage === "Failed to find a default value for discreteFeatureCols")

    // Wrong discreteFeatureCols
    t = intercept[Exception] {
      new FMFeatures()
        .setStrategy(FMFeatures.LRFM2)
        .setContinuousFeatureCols(Array("col1", "col2"))
        .setDiscreteFeatureCols(Array("col3", "col5"))
        .setOutputCol("newFeature")
        .setLabelCol("label")
        .fit(inputDF)
    }
    assert(t.getMessage.contains("Field \"col5\" does not exist."))

    // No continuousFeatureCols
    t = intercept[Exception] {
      new FMFeatures()
        .setStrategy(FMFeatures.LRFM2)
        .setDiscreteFeatureCols(Array("col3", "col4"))
        .fit(inputDF)
    }
    assert(t.getMessage === "Failed to find a default value for continuousFeatureCols")

    // Wrong continuousFeatureCols
    t = intercept[Exception] {
      new FMFeatures()
        .setStrategy(FMFeatures.LRFM2)
        .setContinuousFeatureCols(Array("col1", "col2", "col6"))
        .setDiscreteFeatureCols(Array("col3", "col4"))
        .setOutputCol("newFeature")
        .setLabelCol("label")
        .fit(inputDF)
    }
    assert(t.getMessage.contains("Field \"col6\" does not exist."))

    // No label Col
    t = intercept[Exception] {
      new FMFeatures()
        .setStrategy(FMFeatures.LRFM2)
        .setContinuousFeatureCols(Array("col1", "col2"))
        .setDiscreteFeatureCols(Array("col3", "col4"))
        .fit(inputDF.drop("label"))
    }
    assert(t.getMessage.contains("Field \"label\" does not exist."))

    // Wrong label Col
    t = intercept[Exception] {
      new FMFeatures()
        .setStrategy(FMFeatures.LRFM2)
        .setContinuousFeatureCols(Array("col1", "col2"))
        .setDiscreteFeatureCols(Array("col3", "col4"))
        .setOutputCol("newFeature")
        .setLabelCol("aaa")
        .fit(inputDF)
    }
    assert(t.getMessage.contains("Field \"aaa\" does not exist."))

    // Wrong output Col
    t = intercept[Exception] {
      new FMFeatures()
        .setStrategy(FMFeatures.LRFM2)
        .setContinuousFeatureCols(Array("col1", "col2"))
        .setDiscreteFeatureCols(Array("col3", "col4"))
        .setOutputCol("col1")
        .setLabelCol("label")
        .fit(inputDF)
        .transform(inputDF)
    }
    assert(t.getMessage === "Output column col1 already exists.")

    // invalid strategy type
    t = intercept[Exception] {
      new FMFeatures()
        .setStrategy("aaa")
        .setContinuousFeatureCols(Array("col1", "col2"))
        .setDiscreteFeatureCols(Array("col3", "col4"))
        .setOutputCol("newFeature")
        .setLabelCol("label")
        .fit(inputDF)
    }
    assert(t.getMessage.contains("parameter strategy given invalid value aaa."))

    // invalid numBuckets value
    t = intercept[Exception] {
      new FMFeatures()
        .setStrategy(FMFeatures.LRFM2)
        .setNumBuckets(1)
        .setContinuousFeatureCols(Array("col1", "col2"))
        .setDiscreteFeatureCols(Array("col3", "col4"))
        .setOutputCol("newFeature")
        .setLabelCol("label")
        .fit(inputDF)
    }
    assert(t.getMessage.contains("parameter numBuckets given invalid value 1."))
  }

  test("Test for schema && outcome") {
    val fmFeaturesModel = new FMFeatures()
      .setStrategy(FMFeatures.LRFM2)
      .setContinuousFeatureCols(Array("col1", "col2"))
      .setDiscreteFeatureCols(Array("col3", "col4"))
      .setOutputCol("newFeatures")
      .setLabelCol("label")
      .fit(inputDF)
    val resultSchema = fmFeaturesModel.transformSchema(inputDF.schema)
    val outCome = fmFeaturesModel.transform(inputDF)

    // validate schema
    val outputColName = resultSchema.toArray.map(_.name)
    assert(outputColName === Array("col1", "col2", "col3", "col4", "label", "newFeatures"))

    // The length of ContinuousFeatureCols == the length of OutputColVector
    val lengthArr = outCome.select("newFeatures").collect().map {
      case Row(v: Vector) => v.size
    }
    assert(lengthArr.forall(_ == 2))
  }

  test("Test LRFM1 strategy for FMFeatures") {
    val outCome = FMFeaturesTest.getFMFeatures()
      .setStrategy(FMFeatures.LRFM1).fit(inputDF).transform(inputDF)
    val result = outCome.select("newFeatures").collect().map {
      case Row(v: Vector) => v.toArray
    }
    assert(result.forall(_.length == 1))
  }

  test("Test LRFM2 strategy for FMFeatures") {
    val outCome = FMFeaturesTest.getFMFeatures()
      .setStrategy(FMFeatures.LRFM2).fit(inputDF).transform(inputDF)
    val result = outCome.select("newFeatures").collect().map {
      case Row(v: Vector) => v.toArray
    }
    assert(result.forall(_.length == 2))
  }

  test("Test LRFM3 strategy for FMFeatures") {
    val outCome = FMFeaturesTest.getFMFeatures()
      .setStrategy(FMFeatures.LRFM3).fit(inputDF).transform(inputDF)
    val result = outCome.select("newFeatures").collect().map {
      case Row(v: Vector) => v.toArray
    }
    assert(result.forall(_.length == 1))

  }

  test("Test LRFM4 strategy for FMFeatures") {
    val outCome = FMFeaturesTest.getFMFeatures()
      .setStrategy(FMFeatures.LRFM4).fit(inputDF).transform(inputDF)
    val result = outCome.select("newFeatures")
    val arr = result.collect().map {
      case Row(v: Vector) =>
        v.toArray
    }
    assert(arr.forall(_.length == 2))
  }
}

object FMFeaturesTest {
  def generateTestData(spark: SparkSession): DataFrame = {
    import spark.implicits._
    val testDF = Seq(
      (1.0, 8.0, 1.0, 1.0, 1.0),
      (2.0, 7.0, 1.0, 0.0, 1.0),
      (3.0, 6.0, 0.0, 0.0, 0.0),
      (4.0, 5.0, 1.0, 1.0, 0.0),
      (5.0, 4.0, 1.0, 1.0, 1.0),
      (6.0, 3.0, 1.0, 1.0, 1.0),
      (7.0, 2.0, 0.0, 0.0, 0.0),
      (8.0, 1.0, 0.0, 0.0, 0.0)
    ).toDF("col1", "col2", "col3", "col4", "label")
    testDF
  }
  def getFMFeatures(): FMFeatures ={
    new FMFeatures()
      .setSolver("sgda")
      .setMaxIter(30)
      .setInitialStd(0.2)
      .setUseBiasTerm(false)
      .setUseLinearTerms(false)
      .setNumFactors(3)
      .setRegParam0(0)
      .setRegParam1(0)
      .setRegParam2(0)
      .setTol(1e-7)
      .setThreshold(0.6)
      .setMinBatchFraction(1.0)
      .setStepSize(0.15)
      .setValidationFraction(0.3)
      .setThresholdLR(0.4)
      .setMaxIterLR(20)
      .setTolLR(1e-9)
      .setRegParamLR(0.0)
      .setElasticNetParamLR(0.0)
      .setFitInterceptLR(true)
      .setStandardizationLR(true)
      .setAggregationDepthLR(2)
      .setContinuousFeatureCols(Array("col1", "col2"))
      .setDiscreteFeatureCols(Array("col3", "col4"))
      .setOutputCol("newFeatures")
      .setLabelCol("label")
  }
}
