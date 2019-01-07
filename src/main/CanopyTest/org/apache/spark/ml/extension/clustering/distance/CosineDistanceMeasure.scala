package org.apache.spark.ml.extension.clustering.distance

import org.apache.spark.ml.linalg.BLAS._
import org.apache.spark.ml.linalg._

/**
  * This class implements a cosine distance metric by dividing the dot product of two vectors by
  * the product of their lengths.  That gives the cosine of the angle between the two vectors.
  * To convert this to a usable distance, 1-cos(angle) is what is actually returned.
  */
// 使用余弦距离时，请确保初始向量都已经单位化，否者计算结果会出现错误
class CosineDistanceMeasure extends DistanceMeasure {

  /**
    * @return the distance between two points.
    */
  override def distance(v1: Vector, v2: Vector): Double = {
    fastDistance(new VectorWithNorm(v1), new VectorWithNorm(v2))
  }

  /**
    * If you have special skills, use these skills to calculate the distance.
    *
    * @return the distance between two points.
    */
  override def fastDistance(v1: VectorWithNorm, v2: VectorWithNorm): Double = {
    val vector1 = v1.vector
    val vector2 = v2.vector
    require(vector1.size == vector2.size, s"Vector dimensions do not match: Dim(vector1)" +
      s"=${vector1.size} and Dim(vector1)=${vector2.size}.")

    var denominator = v1.norm * v2.norm
    val dotProduct = dot(vector1, vector2)

    // correct for floating-point rounding errors
    if (denominator < dotProduct) denominator = dotProduct

    // correct for zero-vector corner case
    if (denominator == 0 && dotProduct == 0){
      0.0
    } else {
      1.0 - dotProduct / denominator
    }
  }

  def unitization(v: Vector): Vector = {
    val vector = v.copy
    scal(1.0 / Vectors.norm(vector, 2.0), vector)
    vector
  }

}
