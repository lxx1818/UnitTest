package org.apache.spark.ml.extension.clustering

import com.test.CanopyPressureTest.Ciphertext
import org.apache.spark.ml.extension.clustering.distance.{CosineDistanceMeasure, DistanceMeasure, VectorWithNorm}
import org.apache.spark.ml.linalg.BLAS._
import org.apache.spark.ml.linalg._
import org.apache.spark.ml.param.shared.{HasFeaturesCol, HasPredictionCol}
import org.apache.spark.ml.param.{DoubleParam, Param, ParamMap, ParamValidators}
import org.apache.spark.ml.util.{Identifiable, SchemaUtils}
import org.apache.spark.ml.{Estimator, Model}
import org.apache.spark.sql.functions.{col, udf}
import org.apache.spark.sql.types.{IntegerType, StructType}
import org.apache.spark.sql.{DataFrame, Dataset, Row}
import org.apache.spark.storage.StorageLevel

/**
  * Canopy clustering algorithm:
  * The canopy clustering algorithm is an unsupervised pre-clustering algorithm introduced
  * by Andrew McCallum, Kamal Nigam and Lyle Ungar in 2000.[1] It is often used as preprocessing
  * step for the K-means algorithm or the Hierarchical clustering algorithm. It is intended to
  * speed up clustering operations on large data sets, where using another algorithm directly
  * may be impractical due to the size of the data set.
  *
  * The algorithm proceeds as follows, using two thresholds T1 (the loose distance) and T2 (the
  * tight distance), where T1 > T2.
  *     1.Begin with the set of data points to be clustered.
  *     2.Remove a point from the set, beginning a new 'canopy' containing this point.
  *     3.For each point left in the set, assign it to the new canopy if its distance to the
  *       first point of the canopy is less than the loose distance T1.
  *     4.If the distance of the point is additionally less than the tight distance T2, remove it
  *       from the original set.
  *     5.Repeat from step 2 until there are no more data points in the set to cluster.
  *     6.These relatively cheaply clustered canopies can be sub-clustered using a more expensive
  *       but accurate algorithm.
  *
  * Here is a distributed implementation of the canopy algorithm, refer to the map-reduce
  * implementation strategy in mahout.
  *
  * Strategy for parallelization
  * Looking at the sample Hadoop implementation in http://code.google.com/p/canopy-clustering/
  * the processing is done in 3 M/R steps:
  *     1.The data is massaged into suitable input format.
  *     2.Each mapper performs canopy clustering on the points in its input set and outputs its
  *       canopies’ centers
  *     3.The reducer clusters the canopy centers to produce the final canopy centers
  *     4.The points are then clustered into these final canopies
  *
  * Some ideas can be found in Cluster computing and MapReduce lecture video series [by Google
  * (r)][https://www.youtube.com/watch?v=yjPBkvYh-ss&list=PLEFAB97242917704A];
  * Canopy Clustering is discussed in lecture #4 [https://www.youtube.com/watch?v=1ZDybXl212Q].
  * Finally here is the Wikipedia page[https://en.wikipedia.org/wiki/Canopy_clustering_algorithm].
  */

private[clustering] trait CanopyParams extends HasFeaturesCol with HasPredictionCol {

  // the T1 distance threshold
  final val t1: DoubleParam = new DoubleParam(this, "t1",
    "the T1 distance threshold (>= 0)", ParamValidators.gtEq(0))

  /** @group getParam */
  def getT1: Double = $(t1)

  // the T2 distance threshold
  final val t2: DoubleParam = new DoubleParam(this, "t2",
    "the T2 distance threshold (>= 0)", ParamValidators.gtEq(0))

  /** @group getParam */
  def getT2: Double = $(t2)

  // the T3 distance threshold
  final val t3: DoubleParam = new DoubleParam(this, "t3",
    "the T3 distance threshold (>= 0)", ParamValidators.gtEq(0))

  /** @group getParam */
  def getT3: Double = $(t3)

  // the T4 distance threshold
  final val t4: DoubleParam = new DoubleParam(this, "t4",
    "the T4 distance threshold (>= 0)", ParamValidators.gtEq(0))

  /** @group getParam */
  def getT4: Double = $(t4)

  // The minimum threshold for forming canopy, and can be used for the judgment of
  // isolated points.
  final val canopyThreshold: DoubleParam = new DoubleParam(this, "canopyThreshold",
    "The minimum threshold for forming canopy (>= 0)", ParamValidators.gtEq(0))

  /** @group getParam */
  def getCanopyThreshold: Double = $(canopyThreshold)

  final val distanceMeasure: Param[String] = new Param[String](this, "distanceMeasure",
    "Distance metric between two points. Supported options: " +
      s"${DistanceMeasure.supportedDistanceMeasure.mkString(", ")}. " +
      s"(Default ${DistanceMeasure.SQUAREDEUCLIDEAN})",
    ParamValidators.inArray[String](DistanceMeasure.supportedDistanceMeasure))

  def getDistanceMeasure: String = $(distanceMeasure)

  /**
    * Param for canopies column name.
    * @group param
    */
  final val canopiesCol: Param[String] = new Param[String](this, "canopiesCol",
    "canopies column name")

  /** @group getParam */
  def getCanopiesCol: String = $(canopiesCol)

  setDefault(t1 -> 2.0, t2 -> 1.0, t3 -> 0.0, t4 -> 0.0, canopyThreshold -> 0.0,
    distanceMeasure -> DistanceMeasure.SQUAREDEUCLIDEAN, canopiesCol -> "")

  /** Validates and transforms the input schema. */
  protected def validateAndTransformSchema(schema: StructType): StructType = {
    SchemaUtils.checkColumnType(schema, $(featuresCol), new VectorUDT)
    require(!schema.fieldNames.contains($(predictionCol)),
      s"prediction column ${$(predictionCol)} already exists.")
    require(!schema.fieldNames.contains($(canopiesCol)),
      s"prediction column ${$(predictionCol)} already exists.")

    val newSchema = SchemaUtils.appendColumn(schema, $(canopiesCol), new VectorUDT)
    SchemaUtils.appendColumn(newSchema, $(predictionCol), IntegerType)
  }
}

class Canopy(override val uid: String) extends Estimator[CanopyModel]
  with CanopyParams {

  def this() = this(Identifiable.randomUID("canopy"))

  def setT1(value: Double): this.type = set(t1, value)
  def setT2(value: Double): this.type = set(t2, value)
  def setT3(value: Double): this.type = set(t3, value)
  def setT4(value: Double): this.type = set(t4, value)
  def setCanopyThreshold(value: Double): this.type = set(canopyThreshold, value)
  def setDistanceMeasure(value: String): this.type = set(distanceMeasure, value)
  def setFeaturesCol(value: String): this.type = set(featuresCol, value)
  def setPredictionCol(value: String): this.type = set(predictionCol, value)
  def setCanopiesCol(value: String): this.type = set(canopiesCol, value)

  override def fit(dataset: Dataset[_]): CanopyModel = {
    transformSchema(dataset.schema, logging = true)

    // t3, t4如果没被设置，则默认使用t1, t2的值(t3=t1, t4=t2)
    if ($(t3) == 0.0) setT3($(t1))
    if ($(t4) == 0.0) setT4($(t2))

    require($(t1) > $(t2), s"The value of T1 must be greater than the value of T2")
    require($(t3) > $(t4), s"The value of T3 must be greater than the value of T4")

    val handlePersistence = dataset.storageLevel == StorageLevel.NONE
    val measure = DistanceMeasure.decodeFromString($(distanceMeasure))

    val dataFeatures = dataset.select(col($(featuresCol))).rdd.map { case Row(point: Vector) =>
      measure match {
        // 余弦距离需要单位化向量
        case cosine: CosineDistanceMeasure => cosine.unitization(point)
        case _ => point
      }
    }

    // 将数据持久化，提高迭代计算速度
    if (handlePersistence) {
      dataFeatures.persist(StorageLevel.MEMORY_AND_DISK)
    }
    val iterationStartTime = System.nanoTime()

    var canopyStates = new Array[CanopyState](0)

    // canopy 聚类主要逻辑
    val distributedResult = dataFeatures.mapPartitions { points =>
      // 局部聚类阶段，使用t1,t2聚类
      Canopy.createCanopies($(t1), $(t2), points, measure)
        .map(_.updateCenter()).map(x => (x.center, x.weight)).iterator
    }.collect()

    println(Ciphertext + "map()\t" + (System.nanoTime() - iterationStartTime) / 1e9)

    // 全局聚类阶段，使用t3,t4聚类
    distributedResult.foreach { case (point, weight) =>
      if (canopyStates.length == 0) {
        canopyStates = canopyStates ++ Array(new CanopyState(point, weight))
      } else {
        canopyStates = Canopy.addPointToCanopies($(t3), $(t4), weight,
          point, canopyStates, measure)
      }
    }

    // 还原数据状态
    if (handlePersistence) {
      dataFeatures.unpersist()
    }

    canopyStates = canopyStates.filter(_.weight > $(canopyThreshold)) ++
      canopyStates.filterNot(_.weight > $(canopyThreshold))

    val centers = canopyStates.map(_.updateCenter()).map(x => (x.center, x.weight))

    println(Ciphertext + "fit()\t" + (System.nanoTime() - iterationStartTime) / 1e9)
    println(Ciphertext + "centers\t" + centers.length.toString)

    copyValues(new CanopyModel(uid, centers).setParent(this))
  }

  override def copy(extra: ParamMap): Canopy = defaultCopy(extra)

  override def transformSchema(schema: StructType): StructType = validateAndTransformSchema(schema)
}

object Canopy {

  def createCanopies(
    t1: Double,
    t2: Double,
    points: Iterator[Vector],
    measure: DistanceMeasure): Array[CanopyState] = {
    var canopyStates = new Array[CanopyState](0)

    // Iterate through the points, generate new canopies. Return the canopies.
    points.foreach { point =>
      if (canopyStates.length == 0) {
        canopyStates = canopyStates ++ Array(new CanopyState(point))
      } else {
        canopyStates = addPointToCanopies(t1, t2, 1.0, new VectorWithNorm(point),
          canopyStates, measure)
      }
    }

    canopyStates
  }

  def addPointToCanopies(
      t1: Double,
      t2: Double,
      weight: Double,
      point: VectorWithNorm,
      canopies: Array[CanopyState],
      measure: DistanceMeasure): Array[CanopyState] = {
    // point是否与某个中心点距离小于t2
    var pointStronglyBound = false

    // 更新canopies
    val newCenters = canopies.map { canopy =>
      val dist = measure.fastDistance(canopy.center, point)
      pointStronglyBound = pointStronglyBound || dist < t2
      if (dist < t1) {
        canopy.addPoint(point.vector, weight)
      } else {
        canopy
      }
    }
    // 根据pointStronglyBound判断是否需要创建新的canopy
    if (pointStronglyBound) {
      newCenters
    } else {
      newCenters ++ Array(new CanopyState(point, weight))
    }
  }

}

/**
  * @param centers canopySate的数组——依据weight值降序的有序数组
  */
class CanopyModel(override val uid: String, val centers: Array[(VectorWithNorm, Double)])
  extends Model[CanopyModel] with CanopyParams {

  override def copy(extra: ParamMap): CanopyModel = {
    val copied = new CanopyModel(uid, centers)
    copyValues(copied, extra).setParent(parent)
  }

  def getCenters: Array[Vector] = {
    centers.filter(_._2 > $(canopyThreshold)).map(_._1.vector)
  }

  def getNumCanopies: Int = centers.count(_._2 > $(canopyThreshold))

  // 算法会根据canopyThreshold的值，将权重小于canopyThreshold的canopy的点的预测结果预测为-1。
  override def transform(dataset: Dataset[_]): DataFrame = {
    transformSchema(dataset.schema, logging = true)
    val measure = DistanceMeasure.decodeFromString($(distanceMeasure))

    val distancesUDF = udf((vector: Vector) => getDistacces(vector, measure))
    val distancesCol = $(predictionCol) + "Distances"
    var outputData = dataset.withColumn(distancesCol, distancesUDF(col($(featuresCol))))

    // 生成canopiesCol列,依据t1作为距离
    if($(canopiesCol) != "") {
      val numCanopies = getNumCanopies
      val canopiesUDF = udf((distances: Vector) => canopies(distances, numCanopies))
      outputData = outputData.withColumn($(canopiesCol), canopiesUDF(col(distancesCol)))
    }
    // 生成predictionCol列
    val predictUDF = udf((distances: Vector) => predict(distances))
    outputData = outputData.withColumn($(predictionCol), predictUDF(col(distancesCol)))

    outputData.drop(distancesCol)
  }

  private[clustering] def predict(distances: Vector): Int = {
    val arrDis = distances.toArray
    val index = arrDis.indexOf(arrDis.min)
    // 离样本最近的center必须大于阈值canopyThreshold，否则该样本不予聚类——被认为是孤立点，返回-1。
    if (centers(index)._2 > $(canopyThreshold)) {
      index
    } else {
      -1
    }
  }

  private[clustering] def canopies(distances: Vector, numCanopies: Int): Vector = {
    val canopies = Array.fill(numCanopies)(0.0)
    canopies.indices.foreach { index =>
      if (distances(index) < $(t1)) canopies(index) = 1.0
    }
    Vectors.dense(canopies)
  }

  private[clustering] def getDistacces(
      point: Vector,
      measure: DistanceMeasure): Vector = {
    val pointVWN = new VectorWithNorm(point)
    Vectors.dense(centers.map(center => measure.fastDistance(center._1, pointVWN)))
  }

  override def transformSchema(schema: StructType): StructType = validateAndTransformSchema(schema)

}

private[clustering] class CanopyState(
  var center: VectorWithNorm,
  var sumVector: DenseVector,
  var weight: Double) {

  def this(center: VectorWithNorm, weight: Double) = {
    this(center, CanopyState.generateSumVector(center, weight), weight)
  }

  def this(center: Vector) = this(new VectorWithNorm(center), center.toDense, 1.0)

  def addPoint(point: Vector, weight: Double): this.type = {
    axpy(weight, point, sumVector)
    this.weight += weight
    this
  }

  def updateCenter(): this.type = {
    scal(1.0 / weight, sumVector)
    center = new VectorWithNorm(sumVector.copy)
    this
  }

}

private[clustering] object CanopyState {
  def generateSumVector(center: VectorWithNorm, weight: Double): DenseVector = {
    val sumVector = center.vector.copy.toDense
    scal(weight, sumVector)
    sumVector
  }
}
