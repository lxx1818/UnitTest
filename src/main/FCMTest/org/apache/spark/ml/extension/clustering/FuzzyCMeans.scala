package org.apache.spark.ml.extension.clustering

import breeze.linalg.{DenseVector => BDV, Vector => BV}
import org.apache.spark.broadcast.Broadcast
import org.apache.spark.ml.extension.clustering.distance.VectorWithNorm
import org.apache.spark.mllib.EuclideanDistance
import org.apache.spark.ml.linalg._
import org.apache.spark.ml.param._
import org.apache.spark.ml.param.shared._
import org.apache.spark.ml.util.{Identifiable, SchemaUtils}
import org.apache.spark.ml.{Estimator, Model}
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.functions.{col, udf}
import org.apache.spark.sql.types.{StructField, StructType}
import org.apache.spark.sql.{DataFrame, Dataset, Row}
import org.apache.spark.storage.StorageLevel
import org.apache.spark.util.random.XORShiftRandom


/**
  * FCM（Fuzzy C-means clustering）,模糊C均值聚类
  *
  * FCM（Fuzzy C-means clustering）算法首先是由E. Ruspini提出来的，后来J. C. Dunn与J. C. Bezdek将
  * E. Ruspini算法从硬聚类算法推广成模糊聚类算法。
  *
  * FCM算法是一种无监督的模糊聚类方法，在算法实现过程中不需要人为的干预。FCM算法将n个向量xj分为c个模糊类，并
  * 计算每个聚类的中心，使得非相识性指标达到最小。FCM与硬聚类的主要区别在于FCM采样模糊划分，使得每个给定数据
  * 点xj在[0,1]区间的隶属度uij来确定其属于各个聚类的程度（一个数据集的隶属度的和总等于1）。
  *
  * FCM算法和k-means算法十分相近：
  * 1.设置聚类数目c（或则k）和超参数m（用以控制聚类模糊程度，越高越模糊）
  * 2.给出初始隶属度矩阵U，并赋予初值（满足每个样本属于各类的隶属度和为1）
  * 3.迭代计算直到收敛(前后两次聚类中心或者隶属度变化小于某个阈值)
  *     1）计算新的聚类中心;
  *     2）计算新的隶属度矩阵；
  * 4.输出聚类中心和隶属度矩阵
  *
  * 由于FCM每次迭代都需要大量计算来更新隶属度矩阵和聚类中心点，尤其大数据量和高纬度数据时，每次迭代都需要花费大量
  * 时间，所以为了解决这一问题，人们提出了分布式计算的解决策略——本代码实现了基于spark的FCM分布式算法，极大加快了
  * FCM的运行时间。
  *
  * 本代码算法主要逻辑：
  * 1.初始化：初始化聚类中心点（这里采用从数据集中随机采样k个样本）而不是隶属度矩阵，并加上[0,1]的随机小数——减少
  * 初始聚类中心中出现相同的中心点的概率。隶属度矩阵由聚类中心点和数据集计算得出。
  * 2.迭代直到满足停止条件：
  *     1)将原中心点发送到每个计算节点；
  *     2)map：每个计算节点在每个RDD中计算出各个节点的隶属度向量（vector（u1j, u2j, u3j...ukj），k为中心点数
  *     量）以及隶属度和当前数据点的乘积（array（u1j*xj, u2j*xj, u3j*xj...ukj*xj），xj为第j个样本向量），
  *     组合成元组形式 => (i, uij*xj, uij)，i代表不同的聚类中心点，为key值;
  *     3)reduce：通过key值求和 => SUM(uij*xj)，遍历j求和； SUM(uij)，遍历j求和；
  *     4)新的中心点 ci = (SUM(uij*xj)，遍历j求和) / (SUM(uij)，遍历j求和)
  * 3.输出聚类中心。
  *
  * 参考论文：
  * 1.《FCM: The fuzzy c -means clustering algorithm》，这篇论文由J. C. Bezdek等人所著，主要讲述
  * FCM算法的FORTRAN-IV编码实现，属于单机版，可参考看FCM算法的具体逻辑思路。
  * 网址：[[https://doi.org/10.1016/0098-3004(84)90020-7]]
  * 2.《MapReduce-based fuzzy c-means clustering algorithm: implementation and scalability》，
  * 该文给出了基于mapReduce的FCM算法实现，下面的代码具体实现就是参考了该问的基础上改进得到。
  * 网址：[[https://link.springer.com/article/10.1007/s13042-015-0367-0]]
  *
  */

class FuzzyCMeans(override val uid: String) extends Estimator[FuzzyCMeansModel]
  with FuzzyCMeansParams {
  def this() = this(Identifiable.randomUID("fuzzycmeans"))

  def setK(value: Int): this.type = set(k, value)

  def setM(value: Double): this.type = set(m, value)

  def setMaxIter(value: Int): this.type = set(maxIter, value)

  def setTol(value: Double): this.type = set(tol, value)

  def setSeed(value: Long): this.type = set(seed, value)

  def setFeaturesCol(value: String): this.type = set(featuresCol, value)

  def setPredictionCol(value: String): this.type = set(predictionCol, value)

  override def fit(dataset: Dataset[_]): FuzzyCMeansModel = {

    transformSchema(dataset.schema, logging = true)
    val sc = dataset.sparkSession.sparkContext

    val handlePersistence = dataset.storageLevel == StorageLevel.NONE
    // Compute squared norms and cache them.
    val dataFeatures = dataset.select(col($(featuresCol))).rdd.map {
      case Row(point: Vector) => point
    }

    // 将数据持久化，提高迭代计算速度
    if (handlePersistence) {
      dataFeatures.persist(StorageLevel.MEMORY_AND_DISK)
    }

    val norms = dataFeatures.map { point: Vector => Vectors.norm(point, 2.0) }

    // 将数据持久化，提高迭代计算速度
    norms.persist()

    val zippedData = dataFeatures.zip(norms).map { case (v, norm) =>
      new VectorWithNorm(v, norm)
    }

    val initStartTime = System.nanoTime()

    // 生成初始聚类中心点
    var centers = FuzzyCMeans.initRandomCenters($(k), $(seed), zippedData)
    val initTimeInSeconds = (System.nanoTime() - initStartTime) / 1e9
    logInfo(f"Initialization with random took $initTimeInSeconds%.3f seconds.")

    var converged = false
    var iteration = 0
    // dataDim: 样本维度
    val dataDim = zippedData.first().vector.size
    val iterationStartTime = System.nanoTime()

    // 进行迭代计算
    // 停止条件：算法的迭代次数超过最大迭代次数，或者中心点收敛
    while (iteration < $(maxIter) && !converged) {
      val bcCenters = sc.broadcast(centers)

      // 计算新的聚类中心点
      val newCenters = getNewCenters(zippedData, dataDim, bcCenters)
      bcCenters.destroy(blocking = false)

      // 判断聚类中心点是否收敛，并更新中心点
      converged = newCenters.forall { case (index, newCenter) =>
        FuzzyCMeans.fastSquaredDistance(newCenter, centers(index)) < $(tol) * $(tol)
      }
      centers = newCenters.map(_._2)

      // 迭代次数加一
      iteration += 1
    }

    val iterationTimeInSeconds = (System.nanoTime() - iterationStartTime) / 1e9
    logInfo(f"Iterations took $iterationTimeInSeconds%.3f seconds.")

    if (iteration == $(maxIter)) {
      logInfo(s"FCM reached the max number of iterations: $iteration.")
    } else {
      logInfo(s"FCM converged in $iteration iterations.")
    }

    // 还原数据状态
    norms.unpersist()
    if (handlePersistence) {
      dataFeatures.unpersist()
    }

    copyValues(new FuzzyCMeansModel(uid, centers).setParent(this))
  }

  override def copy(extra: ParamMap): FuzzyCMeans = defaultCopy(extra)

  override def transformSchema(schema: StructType): StructType = validateAndTransformSchema(schema)

  // 计算新的聚类中心点
  private def getNewCenters(
      data: RDD[VectorWithNorm],
      dataDim: Int,
      bcCenters: Broadcast[Array[VectorWithNorm]]): Array[(Int, VectorWithNorm)] = {

    val newCenters = data.mapPartitions { points =>
      /**
        * sumVectorsUijm(i): sum(uijm*xj)，遍历j求和
        * sumUijms(i): sum(uijm)，遍历j求和
        */
      val thisCenters = bcCenters.value
      val sumVectorsUijm = Array
        .fill(thisCenters.length)(BDV.zeros[Double](dataDim).asInstanceOf[BV[Double]])
      val sumUijms = Array.fill[Double](thisCenters.length)(0.0)

      points.foreach { point =>
        val fourfoldUijs = FuzzyCMeans.getMembership(thisCenters, point, $(m))
          .map(_ * thisCenters.length)
        val tmpUijms = fourfoldUijs.map(math.pow(_, $(m)))
        val uijms = if (tmpUijms.forall(_ >= 0.0)) {
          tmpUijms
        } else {
          Array.fill(thisCenters.length)(1.0 / thisCenters.length)
        }
        val pointsBDV = new BDV(point.vector.toArray)
        uijms.zipWithIndex.foreach { case (uijm, index) =>
          sumVectorsUijm(index) += pointsBDV * uijm
          sumUijms(index) += uijm
        }
      }

      sumVectorsUijm.indices.map { index =>
        (index, (sumVectorsUijm(index), sumUijms(index)))
      }.iterator

    }.reduceByKey { case ((sumVectorsUijm1, sumUijms1), (sumVectorsUijm2, sumUijms2)) =>
      (sumVectorsUijm1 + sumVectorsUijm2, sumUijms1 + sumUijms2)
    }.mapValues { case (sumVector, sumUijm) =>
      if (sumUijm != 0){
        new VectorWithNorm(Vectors.fromBreeze(sumVector / sumUijm))
      } else {
        new VectorWithNorm(Vectors.zeros(dataDim))
      }
    }

    newCenters.collect()
  }

}

object FuzzyCMeans {

  /**
    *
    * @param centers 聚类中心点
    * @param point 数据样本点
    * @param m 柔性参数
    * @return uij隶属度，大小：k×1
    */
  private[clustering] def getMembership(
      centers: Array[VectorWithNorm],
      point: VectorWithNorm,
      m: Double): Array[Double] = {
    /**
      * coincideIndex: centers中与point点重合的点的下标，没有的话则值为-1;
      * distancePow(i): distance(xj,centers(i))的(2/(m-1))次方;
      * uijms(i): 隶属度
      */
    // 当 m->1+或m->PositiveInfinity时，结果可能会超出计算范围，或出现距离为0的情况，不能用除法计算隶属度。
    // 判断数据是否超出计算范围
    val distancePow = centers.map{ center =>
      math.pow(fastSquaredDistance(center, point), 1.0 / (m - 1))
    }
    val isLegal = distancePow.forall(dis => dis > 0 && dis != Double.PositiveInfinity)
    if (isLegal) {
      // 正常计算隶属度
      val totalDistancePow = distancePow.map(1 / _).sum
      distancePow.map { dis => math.pow(dis * totalDistancePow, -1) }
    } else {
      // 重合点的隶属度为1，其他为0
      val uijms = Array.fill(centers.length)(0.0)
      uijms(findClosest(centers, point)._1) = 1.0
      uijms
    }
  }

  /**
    *
    * @param v1 向量一
    * @param v2 向量二
    * @return v1和v2的欧式距离的平方
    */
  private[clustering] def fastSquaredDistance(v1: VectorWithNorm, v2: VectorWithNorm): Double = {
    // Vectors.sqdist(v1.vector, v2.vector)
    EuclideanDistance.getfastSquaredDistance(v1.vector, v1.norm, v2.vector, v2.norm)
  }

  /**
    * 仿写Kmeans里的findClosest()方法
    * Returns the index of the closest center to the given point, as well as the squared distance.
    */
  private[clustering] def findClosest(
      centers: TraversableOnce[VectorWithNorm],
      point: VectorWithNorm): (Int, Double) = {

    var bestDistance = Double.PositiveInfinity
    var bestIndex = 0
    var index = 0

    centers.foreach { center =>
      // Since `\|a - b\| \geq |\|a\| - \|b\||`, we can use this lower bound to avoid unnecessary
      // distance computation.
      var lowerBoundOfSqDist = center.norm - point.norm
      lowerBoundOfSqDist = lowerBoundOfSqDist * lowerBoundOfSqDist
      if (lowerBoundOfSqDist < bestDistance) {
        val distance = fastSquaredDistance(center, point)
        if (distance < bestDistance) {
          bestDistance = distance
          bestIndex = index
        }
      }
      index += 1
    }
    (bestIndex, bestDistance)
  }

  // 初始化聚类中心点
  private def initRandomCenters(
      k: Int,
      seed: Long,
      dataset: RDD[VectorWithNorm]): Array[VectorWithNorm] = {
    // Select without replacement; may still produce duplicates if the data has < k distinct
    // points, so deduplicate the centroids before computing center of the data。
    // 采用随机选取初始中心点方法。中心点选取不当会使结果陷入局部最优的情况。
    val sample = dataset.takeSample(false, k, new XORShiftRandom(seed).nextInt())
    // 添加扰动，减少出现重合点的概率。
    addDisturbance(sample)
  }

  /**
    *
    * @param centers original center point
    * @return new center point
    */
  private def addDisturbance(centers: Array[VectorWithNorm]): Array[VectorWithNorm] = {
    centers.map(vwn => new VectorWithNorm(vwn.vector.toArray.map(_ + math.random * 1e-5)))
  }
}

class FuzzyCMeansModel(override val uid: String, val centers: Array[VectorWithNorm])
  extends Model[FuzzyCMeansModel] with FuzzyCMeansParams {

  def setFeaturesCol(value: String): this.type = set(featuresCol, value)

  def setPredictionCol(value: String): this.type = set(predictionCol, value)

  override def copy(extra: ParamMap): FuzzyCMeansModel = {
    val copied = new FuzzyCMeansModel(uid, centers)
    copyValues(copied, extra).setParent(parent)
  }

  override def transform(dataset: Dataset[_]): DataFrame = {
    transformSchema(dataset.schema, logging = true)
    val predictUDF = udf((vector: Vector) => predict(vector))
    dataset.withColumn($(predictionCol), predictUDF(col($(featuresCol))))
  }

  private[clustering] def predict(point: Vector): Int = {
    FuzzyCMeans.findClosest(centers, new VectorWithNorm(point))._1
  }

  override def transformSchema(schema: StructType): StructType = validateAndTransformSchema(schema)

  /**
   * @return The cluster centers
   */
  def getCenters(): Array[Vector] = centers.map(_.vector)

  /**
    * @return Total number of clusters.
    */
  def clusterCentersNum() : Int = centers.length

  /**
    *
    * @param  dataset Input data
    * @return Returns the membership matrix
    */
  def getMembershipMatrix(dataset: Dataset[_]): DataFrame = {
    import dataset.sparkSession.implicits._
    dataset.select($(featuresCol)).rdd.map { case Row(vector: Vector) =>
        getMembership(vector)
    }.toDF("MembershipMatrix")
  }

  /**
    * @param point sample point
    * @return The degree of membership of the sample point relative to each center point
    */
  def getMembership(point: Vector) : Array[Double] = {
    FuzzyCMeans.getMembership(centers, new VectorWithNorm(point), $(m))
  }
}

private[clustering] trait FuzzyCMeansParams extends HasMaxIter with HasSeed with HasFeaturesCol
  with HasTol with HasPredictionCol {

  /**
    * The number of clusters to create (k). Must be &gt; 1. Note that it is possible for fewer than
    * k clusters to be returned, for example, if there are fewer than k distinct points to cluster.
    * Default: 2.
    */
  final val k = new IntParam(this, "k", "The number of clusters to create. "
    + "Must be > 1.", ParamValidators.gt(1))

  def getK: Int = $(k)

  /**
    * The m is the hyper-parameter that controls how fuzzy the cluster will be. The higher it is,
    * the fuzzier the cluster will be in the end. Must be &gt; 1.
    * Note: In most cases, m ∈ [1.5, 2.5] can produce better results. If m > 20, The result will
    *       be not good.
    * Default: 2.
    */
  final val m = new DoubleParam(this, "m", "Controlling the flexibility of" +
    " the FCM algorithm. Must be > 1.", ParamValidators.gt(1))

  def getM: Double = $(m)

  setDefault(
    k -> 2,
    m -> 2,
    maxIter -> 100,
    tol -> 1e-8,
    seed -> 2)

  /** Validates and transforms the input schema. */
  protected def validateAndTransformSchema(schema: StructType): StructType = {
    SchemaUtils.checkColumnType(schema, $(featuresCol), new VectorUDT)
    require(!schema.fieldNames.contains($(predictionCol)),
      s"prediction column ${$(predictionCol)} already exists.")
    val outputFields = schema.fields :+ StructField($(predictionCol), new VectorUDT, false)
    StructType(outputFields)
  }
}