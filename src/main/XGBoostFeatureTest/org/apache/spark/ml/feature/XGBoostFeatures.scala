package org.apache.spark.ml.feature

import ml.dmlc.xgboost4j.scala.spark.params.BoosterParams
import ml.dmlc.xgboost4j.scala.spark.{XGBoostEstimator, XGBoostModel}

import org.apache.spark.ml.linalg.{VectorUDT, Vectors}
import org.apache.spark.ml.param._
import org.apache.spark.ml.param.shared.{HasFeaturesCol, HasLabelCol, HasOutputCol}
import org.apache.spark.ml.util.{Identifiable, SchemaUtils}
import org.apache.spark.ml.{Estimator, Model}
import org.apache.spark.sql.types.{StructField, StructType}
import org.apache.spark.sql.{DataFrame, Dataset}
import org.apache.spark.sql.functions.{col, udf}

import scala.collection.mutable

/**
  * @author lxx
  *         使用XGBoost为数据集添加新特征
  */

private[feature] trait XGBoostFeaturesParams extends BoosterParams
  with HasFeaturesCol with HasLabelCol with HasOutputCol {

  val objective = new Param[String](this, "objective",
    "objective function choice")
  setDefault(objective, "binary:logistic")
  def getObjective: String = $(objective)

  val numClasses = new IntParam(this, "numClasses", "The number of class")
  setDefault(numClasses, 2)
  def getNumClasses: Int = $(numClasses)

  val round = new IntParam(this, "num_round", "numbers of rounds to train")
  setDefault(round, 100)
  def getRonud: Int = $(round)

  /** Validates and transforms the input schema. */
  protected def validateAndTransformSchema(schema: StructType): StructType = {
    SchemaUtils.checkColumnType(schema, $(featuresCol), new VectorUDT)
    require(!schema.fieldNames.contains($(outputCol)),
      s"Output column ${$(outputCol)} already exists.")
    val outputFields = schema.fields :+ StructField($(outputCol), new VectorUDT, false)
    StructType(outputFields)
  }
}

class XGBoostFeatures(override val uid: String)
  extends Estimator[XGBoostFeaturesModel] with XGBoostFeaturesParams {

  def this() = this(Identifiable.randomUID("xgboostfeatures"))

  // set xgboost params
  def setObjective(value: String): this.type = set(objective, value)
  def setNumClasses(value: Int): this.type = set(numClasses, value)
  def setBoosterType(value: String): this.type = set(boosterType, value)
  def setEta(value: Double): this.type = set(eta, value)
  def setGamma(value: Double): this.type = set(gamma, value)
  def setMaxDepth(value: Int): this.type = set(maxDepth, value)
  def setMinChildWeight(value: Double): this.type = set(minChildWeight, value)
  def setMaxDeltaStep(value: Double): this.type = set(maxDeltaStep, value)
  def setSubSample(value: Double): this.type = set(subSample, value)
  def setColSampleByTree(value: Double): this.type = set(colSampleByTree, value)
  def setColSampleByLevel(value: Double): this.type = set(colSampleByLevel, value)
  def setLambda(value: Double): this.type = set(lambda, value)
  def setAlpha(value: Double): this.type = set(alpha, value)
  def setTreeMethod(value: String): this.type = set(treeMethod, value)
  def setSketchEps(value: Double): this.type = set(sketchEps, value)
  def setScalePosWeight(value: Double): this.type = set(scalePosWeight, value)
  def setSampleType(value: String): this.type = set(sampleType, value)
  def setNormalizeType(value: String): this.type = set(normalizeType, value)
  def setRateDrop(value: Double): this.type = set(rateDrop, value)
  def setSkipDrop(value: Double): this.type = set(skipDrop, value)
  def setRound(value: Int): this.type = set(round, value)
  // set outputCol, featuresCol and LabelCol Parameter
  def setOutPutCol(value: String): this.type = set(outputCol, value)
  def setFeaturesCol(value: String): this.type = set(featuresCol, value)
  def setLabelCol(value: String): this.type = set(labelCol, value)

  override def fit(dataset: Dataset[_]): XGBoostFeaturesModel = {
    val estimator = new XGBoostEstimator(Map[String, Any]())
    val paramMap = Map(
      estimator.round.name -> $ {round},
      estimator.objective.name -> $ {objective},
      estimator.numClasses.name -> $ {numClasses},
      estimator.boosterType.name -> $ {boosterType},
      estimator.eta.name -> $ {eta},
      estimator.gamma.name -> $ {gamma},
      estimator.maxDepth.name -> $ {maxDepth},
      estimator.minChildWeight.name -> $ {minChildWeight},
      estimator.maxDeltaStep.name -> $ {maxDeltaStep},
      estimator.subSample.name -> $ {subSample},
      estimator.colSampleByTree.name -> $ {colSampleByTree},
      estimator.colSampleByLevel.name -> $ {colSampleByLevel},
      estimator.lambda.name -> $(lambda),
      estimator.alpha.name -> $ {alpha},
      estimator.treeMethod.name -> $ {treeMethod},
      estimator.sketchEps.name -> $ {sketchEps},
      estimator.scalePosWeight.name -> $ {scalePosWeight},
      estimator.skipDrop.name -> $ {skipDrop},
      estimator.rateDrop.name -> $ {rateDrop},
      estimator.sampleType.name -> $ {sampleType},
      estimator.normalizeType.name -> $ {normalizeType})
    val xgb = new XGBoostEstimator(paramMap)
      .setFeaturesCol($(featuresCol))
      .setLabelCol($(labelCol))
    val model = xgb.fit(dataset)

    val leafIndexMaps = model.booster.getModelDump().map(extractLeaf)
    // 存在至少两层的树
    assert(leafIndexMaps.exists(_.size > 1),
      throw new IllegalArgumentException("All trained trees have only one node."))
    copyValues(new XGBoostFeaturesModel(model, leafIndexMaps).setParent(this))
  }

  private val leafRegex = """(\d+):leaf=""".r

  // 根据xgboost返回的树模型字符串，返回(所有叶子节点的nodeIndex -> 从0开始递增Index)的Map
  // tree str为下面的形式
  // 0:[f2<2.45000005] yes=1,no=2,missing=1\n\t1:leaf=1.43540668\n\t2:leaf=-0.733496368\n
  // 例如：树
  // 0:[f2<2.45000005] yes=1,no=2,missing=1  根节点(非叶子节点)
  //     1:leaf=1.43540668                   叶节点
  //     2:leaf=-0.733496368                 叶节点
  // 返回：Map(1 -> 0, 2 -> 1)
  private def extractLeaf(treeStr: String): Map[Int, Int] = {
    val m = leafRegex.pattern.matcher(treeStr)
    leafRegex.findAllMatchIn(treeStr).map(_.group(1).toInt).zipWithIndex.toMap
  }

  override def transformSchema(schema: StructType): StructType = {
    validateAndTransformSchema(schema)
  }

  override def copy(extra: ParamMap): Estimator[XGBoostFeaturesModel] = defaultCopy(extra)

}

class XGBoostFeaturesModel(override val uid: String,
                           val xgbModel: XGBoostModel,
                           val leafIndexMaps: Array[Map[Int, Int]])
  extends Model[XGBoostFeaturesModel] with XGBoostFeaturesParams {

  def this(xgbModel: XGBoostModel, leafIndexMaps: Array[Map[Int, Int]]) =
    this(Identifiable.randomUID("xgbFeature"), xgbModel, leafIndexMaps)

  override def transform(dataset: Dataset[_]): DataFrame = {
    val dataSetWithLeafIndex = xgbModel.transformLeaf(dataset)

    val generateFeatureUDF = udf { indexedFeature: mutable.WrappedArray[Double] =>
      val newFeatures = indexedFeature.map(_.toInt).zip(leafIndexMaps).filter(_._2.size > 1)
        .map { case (predictLeaf, leafIndexMap) =>
          val subFeature = Array.fill(leafIndexMap.size)(0.0)
          subFeature(leafIndexMap(predictLeaf)) = 1.0
          subFeature
        }.reduce(_ ++ _)
      Vectors.dense(newFeatures).compressed
    }
    dataSetWithLeafIndex.withColumn($(outputCol), generateFeatureUDF(col("predLeaf")))
  }

  override def transformSchema(schema: StructType): StructType = {
    validateAndTransformSchema(schema)
  }

  override def copy(extra: ParamMap): XGBoostFeaturesModel = {
    val copied = new XGBoostFeaturesModel(xgbModel, leafIndexMaps)
    copyValues(copied, extra).setParent(parent)
  }
}
