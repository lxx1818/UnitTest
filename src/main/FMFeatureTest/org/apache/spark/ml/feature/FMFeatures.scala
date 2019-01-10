package org.apache.spark.ml.feature

import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.ml.fm._
import org.apache.spark.ml.linalg.{Vector, VectorUDT}
import org.apache.spark.ml.param._
import org.apache.spark.ml.param.shared.HasOutputCol
import org.apache.spark.ml.util.{DefaultParamsWritable, Identifiable, SchemaUtils}
import org.apache.spark.ml.{Estimator, Model}
import org.apache.spark.sql.functions.{col, udf}
import org.apache.spark.sql.types.{StructField, StructType}
import org.apache.spark.sql.{DataFrame, Dataset}

/**
  * The introduction of FMFeatures:
  * Transforms the categorical variables with many levels into a few numeric variables
  *
  * Algorithm background:
  * Including categorical variables with many levels in a logistic regression model easily leads
  * to a sparse design matrix. This can result in a big, ill-conditioned optimization problem
  * causing overfitting, extreme coefficient values and long run times. Inspired by recent
  * developments in matrix factorization, we propose four new strategies of overcoming this
  * problem. Each strategy uses a Factorization Machine that transforms the categorical variables
  * with many levels into a few numeric variables that are subsequently used in the logistic
  * regression model.
  *
  * The introduction of strategies:
  * Notation
  * We model a binary target y with help of p numeric variables x1,x2...xp,
  * and q categorical variables d1,d2...dq with l1,l2...lq levels.
  * LRFM1:
  * The model LRFM1 (Logistic Regression with Factorization Machines 1) is the first model
  * showing our new approach. The categorical variables with many levels d1,d2...dq are first
  * put into a Factorization Machine f0 whose cofficients are estimated from a train set with
  * the target variable y. The output of f0——denoted by g0——is then added to the model equation
  * of the logistic regression.
  *                               g0 = f0(d1,d2...dq)
  * LRFM2:
  * Although LRFM1 is able to model interactions between categorical variables with many levels,
  * it does not model interactions between the variables d1,d2...dq and one or more numeric
  * variables xj . For this reason we allow the model equation to be extended with additional
  * variables g1,g2...gt, where t < p. Each gj is a prediction from a Factorization Machine
  * fj that takes as input the categorical variables d1,d2...dq and a variable Xj . The
  * variable Xj is a discretized version of xj.
  *                               gj = fj(Xj,d1,d2...dq);   1 <= j <= p
  * LRFM3:
  * Instead of learning the coefficients of a Factorization Machine f on a train set with known
  * binary target y, we can do an intermediate step. We first fit a logistic regression model
  * with the numeric variables x1,x2...xp on the train set(so without d1,d2...dq), and then
  * compute the deviance residuals r. The residual vector r can then be used to train the
  * coefficients of the Factorization Machine instead of the original target y.
  * LRFM4:
  * Similarly as LRFM1 was extended to LRFM2, we can extend LRFM3 to LRFM4. More speciffically,
  * we form additional variables hj by including a discretized numeric variable Xj in the
  * Factorization Machine that is trained on residuals.
  *
  * This algorithm is completed with reference to the following papers. For more information,
  * please refer to the paper:《Extending Logistic Regression Models with Factorization Machines》
  * URL: https://doi.org/10.1007/978-3-319-60438-1_32
  */

private[ml] trait FMFeaturesParams extends FactorizationMachinesParams with HasOutputCol {

  /**
    * The strategy for generating features by FM.
    * Supported options: LRFM1, LRFM2, LRFM3, LRFM4.
    * Default: "LRFM1"
    * 这些策略都是围绕如何进行离散属性的处理（参见以下论文）
    * https://doi.org/10.1007/978-3-319-60438-1_32
    * 根据论文所述，当离散属性的值的类别个数很大时（大于1000），逻辑回归结果将会有较大的提升
    * 例如性别属性（男or女）类别个数为2，而类似邮编属性类别个数将会很大（处理后效果会更好的提升）
    * @group param
    */
  final val strategy: Param[String] = new Param[String](this,
    "strategy", "The strategy for generating features by FM. Supported options: "
      + s"${FMFeatures.supportedStrategies.mkString(", ")}. (Default LRFM1)",
    ParamValidators.inArray[String](FMFeatures.supportedStrategies))

  /** @group getParam */
  final def getStrategy: String = $(strategy)

  /**
    * Number of buckets (quantiles, or categories) into which data points are grouped. Must
    * be greater than or equal to 2.
    * Used to discretize continuous features in LRFM2 and LRFM4.
    */
  val numBuckets = new IntParam(this, "numBuckets", "Number of buckets (quantiles, or " +
    "categories) into which data points are grouped. Must be >= 2.",
    ParamValidators.gtEq(2))

  /** @group getParam */
  def getNumBuckets: Int = $(numBuckets)

  /**
    * Param for continuous feature column names.
    * @group param
    */
  final val continuousFeatureCols: StringArrayParam = new StringArrayParam(this,
    "continuousFeatureCols", "continuous feature column names")

  /** @group getParam */
  final def getContinuousFeatureCols: Array[String] = $(continuousFeatureCols)

  /**
    * Param for discrete feature column names.
    * @group param
    */
  final val discreteFeatureCols: StringArrayParam = new StringArrayParam(this,
    "discreteFeatureCols", "discrete feature column names")

  /** @group getParam */
  final def getDiscreteFeatureCols: Array[String] = $(discreteFeatureCols)

  /**
    * FM参数
    */
  /**
    * Param for ratio of sampling during Fm training
    * Range is (0, 1].
    * @group param
    */
  final val minBatchFraction : DoubleParam =
    new DoubleParam(this, "minBatchFraction", "minBatchFraction")

  /** @group getParam */
  final def getMinBatchFraction: Double = $(minBatchFraction)

  /**
    * LR参数，由于LR与FM中许多参数重名，故LR所有参数都加后缀“LR”，以便区分。
    */

  /**
    * Param for threshold in binary classification prediction, in range [0, 1].
    * @group param
    */
  val thresholdLR: DoubleParam = new DoubleParam(this, "thresholdLR", "threshold in binary " +
    "classification prediction, in range [0, 1]", ParamValidators.inRange(0, 1))

  /** @group getParam */
  def getThresholdLR: Double = $(thresholdLR)

  /**
    * Param for maximum number of iterations (&gt;= 0).
    * @group param
    */
  final val maxIterLR: IntParam = new IntParam(this, "maxIterLR", "maximum number of iterations " +
    "(>= 0)", ParamValidators.gtEq(0))

  /** @group getParam */
  final def getMaxIterLR: Int = $(maxIterLR)

  /**
    * Param for the convergence tolerance for iterative algorithms (&gt;= 0).
    * @group param
    */
  final val tolLR: DoubleParam = new DoubleParam(this, "tolLR", "the convergence tolerance for " +
    "iterative algorithms (>= 0)", ParamValidators.gtEq(0))

  /** @group getParam */
  final def getTolLR: Double = $(tolLR)

  /**
    * Param for regularization parameter (&gt;= 0).
    * @group param
    */
  final val regParamLR: DoubleParam = new DoubleParam(this, "regParamLR", "regularization " +
    "parameter (>= 0)", ParamValidators.gtEq(0))

  /** @group getParam */
  final def getRegParamLR: Double = $(regParamLR)

  /**
    * Param for the ElasticNet mixing parameter, in range [0, 1]. For alpha = 0, the penalty
    * is an L2 penalty. For alpha = 1, it is an L1 penalty.
    * @group param
    */
  final val elasticNetParamLR: DoubleParam = new DoubleParam(this,
    "elasticNetParamLR", "the ElasticNet mixing parameter, in range [0, 1]. For " +
      "alpha = 0, the penalty is an L2 penalty. For alpha = 1, it is an L1 penalty",
    ParamValidators.inRange(0, 1))

  /** @group getParam */
  final def getElasticNetParamLR: Double = $(elasticNetParamLR)

  /**
    * Param for whether to fit an intercept term.
    * @group param
    */
  final val fitInterceptLR: BooleanParam = new BooleanParam(this, "fitInterceptLR", "whether to " +
    "fit an intercept term")

  /** @group getParam */
  final def getFitInterceptLR: Boolean = $(fitInterceptLR)

  /**
    * Param for whether to standardize the training features before fitting the model.
    * @group param
    */
  final val standardizationLR: BooleanParam = new BooleanParam(this, "standardizationLR",
    "whether to standardize the training features before fitting the model")

  /** @group getParam */
  final def getStandardizationLR: Boolean = $(standardizationLR)

  /**
    * Param for suggested depth for treeAggregate (&gt;= 2).
    * @group expertParam
    */
  final val aggregationDepthLR: IntParam = new IntParam(this, "aggregationDepthLR",
    "suggested depth for treeAggregate (>= 2)", ParamValidators.gtEq(2))

  /** @group expertGetParam */
  final def getAggregationDepthLR: Int = $(aggregationDepthLR)

  /**
    * Param for weight column name. If this is not set or empty, we treat all instance
    * weights as 1.0.
    * @group param
    */
  final val weightColLR: Param[String] = new Param[String](this, "weightColLR",
    "weight column name. If this is not set or empty, we treat all instance weights as 1.0")

  /** @group getParam */
  final def getWeightColLR: String = $(weightColLR)


  setDefault(strategy -> FMFeatures.LRFM1, numBuckets -> 5)
  // Set FM default parameters
  setDefault(solver -> FactorizationMachines.GD, maxIter -> 5, initialStd -> 0.1,
    useBiasTerm -> true, useLinearTerms -> true, numFactors -> 8, regParam0 -> 0,
    regParam1 -> 1e-3, regParam2 -> 1e-4, tol -> 1e-4, threshold -> 0.5, miniBatchFraction -> 1,
    stepSize -> 0.1, validationFraction -> 0.2, minBatchFraction -> 1)
  // Set LR default parameters
  setDefault(thresholdLR -> 0.5, maxIterLR -> 100, tolLR -> 1E-6, regParamLR -> 0.0,
    elasticNetParamLR -> 0.0, fitInterceptLR -> true, standardizationLR -> true,
    aggregationDepthLR -> 2, weightColLR -> "")

  /** Validates and transforms the input schema. */
  protected def validateAndTransformSchema(schema: StructType): StructType = {
    val discreteFeatureColNames = $(discreteFeatureCols)
    val continuousFeatureColNames = $(continuousFeatureCols)
    // 验证离散特征名
    discreteFeatureColNames.foreach(SchemaUtils.checkNumericType(schema, _))
    // 验证连续特征名
    continuousFeatureColNames.foreach(SchemaUtils.checkNumericType(schema, _))
    // 验证label列和outputCol列
    SchemaUtils.checkNumericType(schema, $(labelCol))
    require(!schema.fieldNames.contains($(outputCol)),
      s"Output column ${$(outputCol)} already exists.")

    val outputFields = schema.fields :+ StructField($(outputCol), new VectorUDT, false)
    StructType(outputFields)
  }

}

class FMFeatures (override val uid: String)
  extends Estimator[FMFeaturesModel] with FMFeaturesParams with DefaultParamsWritable {

  def this() = this(Identifiable.randomUID("fmfeatures"))

  // set params
  def setStrategy(value: String): this.type = set(strategy, value)
  def setOutputCol(value: String): this.type = set(outputCol, value)
  def setLabelCol(value: String): this.type = set(labelCol, value)
  def setNumBuckets(value: Int): this.type = set(numBuckets, value)
  def setContinuousFeatureCols(value: Array[String]): this.type = set(continuousFeatureCols, value)
  def setDiscreteFeatureCols(value: Array[String]): this.type = set(discreteFeatureCols, value)
  // set FM params
  def setSolver(value: String): this.type = set(solver, value)
  def setMaxIter(value: Int): this.type = set(maxIter, value)
  def setInitialStd(value: Double): this.type = set(initialStd, value)
  def setUseBiasTerm(value: Boolean): this.type = set(useBiasTerm, value)
  def setUseLinearTerms(value: Boolean): this.type = set(useLinearTerms, value)
  def setNumFactors(value: Int): this.type = set(numFactors, value)
  def setRegParam0(value: Double): this.type = set(regParam0, value)
  def setRegParam1(value: Double): this.type = set(regParam1, value)
  def setRegParam2(value: Double): this.type = set(regParam2, value)
  def setTol(value: Double): this.type = set(tol, value)
  def setThreshold(value: Double): this.type = set(threshold, value)
  def setMinBatchFraction(value: Double): this.type = set(minBatchFraction, value)
  def setStepSize(value: Double): this.type = set(stepSize, value)
  def setValidationFraction(value: Double): this.type = set(validationFraction, value)
  // set LR params
  def setThresholdLR(value: Double): this.type = set(thresholdLR, value)
  def setMaxIterLR(value: Int): this.type = set(maxIterLR, value)
  def setTolLR(value: Double): this.type = set(tolLR, value)
  def setRegParamLR(value: Double): this.type = set(regParamLR, value)
  def setElasticNetParamLR(value: Double): this.type = set(elasticNetParamLR, value)
  def setFitInterceptLR(value: Boolean): this.type = set(fitInterceptLR, value)
  def setStandardizationLR(value: Boolean): this.type = set(standardizationLR, value)
  def setAggregationDepthLR(value: Int): this.type = set(aggregationDepthLR, value)
  def setWeightColLR(value: String): this.type = set(weightColLR, value)

  override def fit(dataset: Dataset[_]): FMFeaturesModel = {
    // 首先进行onehot处理
    val (oneHotFeatureCols, oneHoutDF) = FMFeatures.oneHotFeatures(dataset, $(discreteFeatureCols))

    val fmModels = $(strategy) match {
      case FMFeatures.LRFM1 =>
        // 生成一个FMModel
        generateFMModel(oneHoutDF, $(labelCol), oneHotFeatureCols)

      case FMFeatures.LRFM2 =>
        // 生成多个FMModel
        generateFMModels(oneHoutDF, $(labelCol), oneHotFeatureCols,
          $(continuousFeatureCols))

      case FMFeatures.LRFM3 =>
        // 生成新label列
        val lr = getLogisticRegression()
        val (labelColName, nlDF) = FMFeatures.generateNewLabelByLR(oneHoutDF,
          $(labelCol), $(continuousFeatureCols), lr)
        // 生成一个FMModel
        generateFMModel(nlDF, labelColName, oneHotFeatureCols)

      case FMFeatures.LRFM4 =>
        // 生成新label列
        val lr = getLogisticRegression()
        val (labelColName, nlDF) = FMFeatures.generateNewLabelByLR(oneHoutDF,
          $(labelCol), $(continuousFeatureCols), lr)
        // 生成多个FMModel
        generateFMModels(nlDF, labelColName, oneHotFeatureCols,
          $(continuousFeatureCols))
    }
    copyValues(new FMFeaturesModel(uid, fmModels).setParent(this))
  }

  override def copy(extra: ParamMap): FMFeatures = defaultCopy(extra)

  override def transformSchema(schema: StructType): StructType = {
    validateAndTransformSchema(schema)
  }

  /**
    * Generate a new FM based on user parameters
    * @return a new FM
    */
  private def getFactorizationMachines(): FactorizationMachines = {
    val fm = new FactorizationMachines()
      .setSolver(${solver})
      .setMaxIter(${maxIter})
      .setInitialStd(${initialStd})
      .setUseBiasTerm(${useBiasTerm})
      .setUseLinearTerms(${useLinearTerms})
      .setNumFactors(${numFactors})
      .setRegParam0(${regParam0})
      .setRegParam1(${regParam1})
      .setRegParam2(${regParam2})
      .setTol($(tol))
      .setThreshold(${threshold})
    // 根据strategy类型设置FM的task参数
    $(strategy) match {
      case FMFeatures.LRFM1 => fm.setTask(FactorizationMachines.Classification)
      case FMFeatures.LRFM2 => fm.setTask(FactorizationMachines.Classification)
      case FMFeatures.LRFM3 => fm.setTask(FactorizationMachines.Regression)
      case FMFeatures.LRFM4 => fm.setTask(FactorizationMachines.Regression)
    }
    // 根据solver类型进行参数设置
    $(solver) match {
      case FactorizationMachines.GD =>
        fm.setMiniBatchFraction(${minBatchFraction}).setStepSize(${stepSize})
      case FactorizationMachines.SGDA =>
        fm.setMiniBatchFraction(${minBatchFraction})
          .setStepSize(${stepSize})
          .setValidationFraction($(validationFraction))
      case FactorizationMachines.PGD =>
        fm.setStepSize(${stepSize})
    }
  }

  /**
    * Generate a new LR based on user parameters
    * @return a new LR
    */
  private def getLogisticRegression(): LogisticRegression = {
    val logisticRegression = new LogisticRegression()
      .setThreshold(${thresholdLR})
      .setMaxIter(${maxIterLR})
      .setTol($(tolLR))
      .setRegParam($(regParamLR))
      .setElasticNetParam($(elasticNetParamLR))
      .setFitIntercept($(fitInterceptLR))
      .setStandardization($(standardizationLR))
      .setAggregationDepth(${aggregationDepthLR})
    // 判断是否存在weightColLR列
    if (!$(weightColLR).equals("")) {
      logisticRegression.setWeightCol($(weightColLR))
    }
    logisticRegression
  }

  /**
    * Generate one FM model that will be used to gengrate new featrue in FMFeaturesModel.
    * Used when LRFM1 or LRFM3.
    *
    * @param dataset The dataset
    * @param labelColName The name of label col
    * @param discreteFeatureCols Discrete feature Cols
    * @return one FM model
    */
  private def generateFMModel(
      dataset: Dataset[_],
      labelColName: String,
      discreteFeatureCols: Array[String]): Array[FactorizationMachinesModel] = {
    val dfWithDV = new VectorAssembler()
      .setInputCols(discreteFeatureCols)
      .setOutputCol(FMFeatures.dFVCol)
      .transform(dataset)

    val fm = getFactorizationMachines()

    val fmModel = fm.setFeaturesCol(FMFeatures.dFVCol)
      .setLabelCol(labelColName)
      .setPredictionCol(FMFeatures.newFeatureColName)
      .fit(dfWithDV)

    Array(fmModel)
  }

  /**
    * Generate multiple FM models that will be used to gengrate new featrue in FMFeaturesModel.
    * Used when LRFM2 or LRFM4.
    *
    * @param dataset The dataset
    * @param labelColName The name of label col
    * @param discreteFeatureCols Discrete feature Cols
    * @param continuousFeatureCols Continuous feature cols
    * @return multiple FM models
    */
  private def generateFMModels(
      dataset: Dataset[_],
      labelColName: String,
      discreteFeatureCols: Array[String],
      continuousFeatureCols: Array[String]): Array[FactorizationMachinesModel] = {
    val newCFColsName = continuousFeatureCols.map(_ + "Categorical")

    // 对连续特征列进行分桶离散化
    val resultDF = new QuantileDiscretizer()
      .setInputCols(continuousFeatureCols)
      .setOutputCols(newCFColsName)
      .setNumBucketsArray(Array.fill(newCFColsName.length)($(numBuckets)))
      .fit(dataset).transform(dataset)

    // 对新离散化特征进行oneHot编码
    val (oneHotFeatureCols, oneHoutDF) = FMFeatures.oneHotFeatures(resultDF, newCFColsName)

    // 持久化数据
    oneHoutDF.persist()

    // 结合离散化后的列，生成多个FMModel
    val fMModels = oneHotFeatureCols.map { colName =>
      val newDiscreteFeatureCols = discreteFeatureCols ++ Array(colName)
      generateFMModel(oneHoutDF, labelColName, newDiscreteFeatureCols)
    }.reduce(_ ++ _)

    // 还原数据状态
    oneHoutDF.unpersist()

    fMModels
  }
}

object FMFeatures {

  /** The first strategy for generating features by FM.
    * 用FM直接处理离散型属性（对label进行学习），生成新的特征。具体实现策略参见以下网址中论文的LRFM1策略。
    * https://doi.org/10.1007/978-3-319-60438-1_32
    */
  private[ml] val LRFM1 = "LRFM1"

  /** The Second strategy for generating features by FM.
    * 在LRFM1基础上改进，每次选取一个连续属性并离散化（等频分箱）加入到离散特征中，再用FM生成新特征。
    * 具体实现策略参见以下网址中论文的LRFM2策略。
    * https://doi.org/10.1007/978-3-319-60438-1_32
    */
  private[ml] val LRFM2 = "LRFM2"

  /** The third strategy for generating features by FM.
    * 选取离散属性，用FM对逻辑回归的参差进行学习，生成新特征。具体实现策略参见以下网址中论文的LRFM3策略。
    * https://doi.org/10.1007/978-3-319-60438-1_32
    */
  private[ml] val LRFM3 = "LRFM3"

  /** The fourth strategy for generating features by FM.
    * 在LRFM3基础上改进，每次选取一个连续属性并离散化（等频分箱）加入到离散特征中，再用FM对逻辑回归的参差进
    * 行学习，生成新特征。具体实现策略参见以下网址中论文的LRFM4策略。
    * https://doi.org/10.1007/978-3-319-60438-1_32
    */
  private[ml] val LRFM4 = "LRFM4"

  /** Set of strategies for generating features by FM. */
  private[ml] val supportedStrategies = Array(LRFM1, LRFM2, LRFM3, LRFM4)

  /** Maximum boundary value of deviance residual */
  private[ml] val maxDevianceResidual = 1000

  /** Column name after continuous features vectorization */
  private[ml] val cFVCol = "continuousFeatureVectorCol"

  /** Column name after discrete features vectorization */
  private[ml] val dFVCol = "discreteFeatureVectorCol"

  /** Column name of Newly generated feature column in LRFM1 or LRFM3 */
  private[ml] val newFeatureColName = "newFeatureCol"

  /**
    * Generate the residual col by LR training as a new label column.
    * Used when LRFM3 or LRFM4.
    *
    * @param dataset Training dataset
    * @param labelColName The name of label col
    * @param continuousFeatureCols Continuous feature cols
    * @param lr LogisticRegression
    * @return (The name of residual col, Dataset)
    */
  private[ml] def generateNewLabelByLR(
      dataset: Dataset[_],
      labelColName: String,
      continuousFeatureCols: Array[String],
      lr: LogisticRegression): (String, DataFrame) = {
    val dfWithDV = new VectorAssembler()
      .setInputCols(continuousFeatureCols)
      .setOutputCol(cFVCol)
      .transform(dataset)

    val lrDF = lr
      .setFeaturesCol(cFVCol)
      .setLabelCol(labelColName)
      .fit(dfWithDV)
      .transform(dfWithDV)

    val devianceResidualsUDF = udf((label: Double, probability: Vector) =>
      getDevianceResiduals(label, probability(1)))

    val resultDF = lrDF.withColumn(labelColName + "DR",
      devianceResidualsUDF(col(labelColName), col(lr.getProbabilityCol)))
      .drop(lr.getFeaturesCol, lr.getPredictionCol, lr.getProbabilityCol, lr.getRawPredictionCol)

    (labelColName + "DR", resultDF)
  }

  /**
    * Calculating residuals. Used when LRFM3 or LRFM4.
    *
    * @param label True label value
    * @param probability The predicted probability that label = 1
    * @return The deviance residuals
    */
  private[ml] def getDevianceResiduals(label: Double, probability: Double): Double = {

    if (label == probability) {
      0.0
    } else if (Math.abs(label - probability) == 1.0) {
      Math.pow(-1, label + 1) * maxDevianceResidual
    } else {
      val result = label match {
        case 0 =>
          val value = (1 - label) * (Math.log(1 - label) - Math.log( 1 - probability))
          -1 * Math.pow(2 * value, 0.5)
        case 1 =>
          val value = label * (Math.log(label) - Math.log(probability))
          Math.pow(2 * value, 0.5)
      }

      // 判读是否越界，如果越界，则使用边界值。
      if (result < maxDevianceResidual && result > -1 * maxDevianceResidual) {
        result
      } else if (result >= maxDevianceResidual) {
        maxDevianceResidual
      } else {
        -1 * maxDevianceResidual
      }
    }
  }

  /**
    *
    * @param dataset Training dataset
    * @param featureCols Discrete feature Cols
    * @return (New feature column name, Onehot processed data)
    */
  private[ml] def oneHotFeatures(
      dataset: Dataset[_],
      featureCols: Array[String]): (Array[String], DataFrame) = {
    val oneHotFeatureCols = featureCols.map(_ + "OneHot")
    val resultDF = new OneHotEncoderEstimator()
      .setDropLast(false)
      .setInputCols(featureCols)
      .setOutputCols(oneHotFeatureCols)
      .fit(dataset)
      .transform(dataset)
    (oneHotFeatureCols, resultDF)
  }

}

class FMFeaturesModel (override val uid: String, val fmModels: Array[FactorizationMachinesModel])
  extends Model[FMFeaturesModel] with FMFeaturesParams {

  override def copy(extra: ParamMap): FMFeaturesModel = {
    val fmFeaturesModel = new FMFeaturesModel(uid, fmModels)
    copyValues(fmFeaturesModel, extra).setParent(parent)
  }

  override def transform(dataset: Dataset[_]): DataFrame = {
    // 首先进行onehot处理
    val (oneHotFeatureCols, oneHoutDF) = FMFeatures.oneHotFeatures(dataset, $(discreteFeatureCols))

    val (newFeaturesCols, resultDF) = $(strategy) match {
      case FMFeatures.LRFM1 =>
        generateFMFeature(oneHoutDF, oneHotFeatureCols, fmModels(0))

      case FMFeatures.LRFM2 =>
        generateFMFeatures(oneHoutDF, oneHotFeatureCols,
          $(continuousFeatureCols))

      case FMFeatures.LRFM3 =>
        generateFMFeature(oneHoutDF, oneHotFeatureCols, fmModels(0))

      case FMFeatures.LRFM4 =>
        generateFMFeatures(oneHoutDF, oneHotFeatureCols,
          $(continuousFeatureCols))
    }
    new VectorAssembler().setInputCols(newFeaturesCols).setOutputCol($(outputCol))
      .transform(resultDF).drop(newFeaturesCols: _*).drop(oneHotFeatureCols: _*)
  }

  override def transformSchema(schema: StructType): StructType = {
    validateAndTransformSchema(schema)
  }

  /**
    * Generate one new feature col. Used when LRFM1 or LRFM3.
    *
    * @param dataset The dataset
    * @param discreteFeatureCols Discrete feature Cols
    * @param fmModel FactorizationMachinesModel
    * @return (The name of new Features col, Dataset)
    */
  private def generateFMFeature(
      dataset: Dataset[_],
      discreteFeatureCols: Array[String],
      fmModel: FactorizationMachinesModel): (Array[String], DataFrame) = {
    val dfWithDV = new VectorAssembler()
      .setInputCols(discreteFeatureCols)
      .setOutputCol(FMFeatures.dFVCol)
      .transform(dataset)

    val result = fmModel.transform(dfWithDV)

    (Array(FMFeatures.newFeatureColName), result.drop(FMFeatures.dFVCol))
  }

  /**
    * Generate multiple new feature cols. Used when LRFM2 or LRFM4.
    *
    * @param dataset The dataset
    * @param discreteFeatureCols Discrete feature Cols
    * @param continuousFeatureCols Continuous feature cols
    * @return (The name of new Features cols, Dataset)
    */
  private def generateFMFeatures(
      dataset: Dataset[_],
      discreteFeatureCols: Array[String],
      continuousFeatureCols: Array[String]): (Array[String], DataFrame) = {
    val newFeaturesColsName = continuousFeatureCols.map("newFeatureFrom" + _)
    val newCFColsName = continuousFeatureCols.map(_ + "Categorical")

    // 对连续特征列进行分桶离散化
    val qdDF = new QuantileDiscretizer()
      .setInputCols(continuousFeatureCols)
      .setOutputCols(newCFColsName)
      .setNumBucketsArray(Array.fill(newCFColsName.length)($(numBuckets)))
      .fit(dataset).transform(dataset)

    // 对新离散化特征进行oneHot编码
    val (oneHotFeatureCols, oneHoutDF) = FMFeatures.oneHotFeatures(qdDF, newCFColsName)

    // 持久化数据
    oneHoutDF.persist()

    var resultDF = oneHoutDF
    // 结合离散化后的列和FMModel，生成多列新特征
    oneHotFeatureCols.indices.foreach { index =>
      val newDiscreteFeatureCols = discreteFeatureCols ++ Array(oneHotFeatureCols(index))
      val (colName, df) = generateFMFeature(resultDF, newDiscreteFeatureCols, fmModels(index))
      resultDF = df.withColumnRenamed(colName(0), newFeaturesColsName(index))
    }

    resultDF = resultDF.drop(newCFColsName: _*).drop(oneHotFeatureCols: _*)
    // 还原数据状态
    oneHoutDF.unpersist()

    (newFeaturesColsName, resultDF)
  }
}