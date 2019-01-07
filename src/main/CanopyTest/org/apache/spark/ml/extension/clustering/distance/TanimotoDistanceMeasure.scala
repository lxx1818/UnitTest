package org.apache.spark.ml.extension.clustering.distance

import org.apache.spark.ml.linalg.BLAS._
import org.apache.spark.ml.linalg.Vector

/**
  * Tanimoto coefficient implementation.
  *
  * http://en.wikipedia.org/wiki/Jaccard_index
  */
class TanimotoDistanceMeasure extends DistanceMeasure {

  /**
    * Calculates the distance between two vectors.
    *
    * The coefficient (a measure of similarity) is: T(a, b) = a.b / (|a|^2 + |b|^2 - a.b)
    *
    * The distance d(a,b) = 1 - T(a,b)
    *
    * @return 0 for perfect match, > 0 for greater distance
    */

  /**
    * @return the distance between two points.
    */
  override def distance(v1: Vector, v2: Vector): Double = {
    fastDistance(new VectorWithNorm(v1), new VectorWithNorm(v2))
  }

  override def fastDistance(v1: VectorWithNorm, v2: VectorWithNorm): Double = {
    val vector1 = v1.vector
    val vector2 = v2.vector
    require(vector1.size == vector2.size, s"Vector dimensions do not match: Dim(vector1)" +
      s"=${vector1.size} and Dim(vector1)=${vector2.size}.")

    val dotProduct = dot(vector1, vector2)
    var denominator = v1.norm * v1.norm + v2.norm * v2.norm - dotProduct

    // correct for fp round-off: distance >= 0
    if (denominator < dotProduct) {
      denominator = dotProduct
    }
    if (denominator > 0) {
      // denominator == 0 only when dot(v1,v1) == dot(v2,v2) == dot(v1,v2) == 0
      return 1.0 - dotProduct / denominator
    } else {
      0.0
    }
  }

}
