package org.apache.spark.ml.extension.clustering.distance

import org.apache.spark.ml.linalg.BLAS.axpy
import org.apache.spark.ml.linalg.{Vector, Vectors}

/**
  * Implement Minkowski distance, a real-valued generalization of the
  * integral L(n) distances: Manhattan = L1, Euclidean = L2.
  * For high numbers of dimensions, very high exponents give more useful distances.
  *
  * Note: Math.pow is clever about integer-valued doubles.
  */
class MinkowskiDistanceMeasure extends DistanceMeasure {

  // The exponent used to calculate the Minkowski distance. Default = 2.0
  private var exponent = 2.0

  def setExponent(value: Double): Unit = {
    require(exponent >= 1.0, "To compute the Minkowski distance between two Vector, we require " +
      "that you specify a exponent>=1. " + s"You specified exponent=$exponent.")
    this.exponent = value
  }

  def getExponent: Double = this.exponent

  /**
    * @return the distance between two points.
    */
  override def distance(v1: Vector, v2: Vector): Double = {
    require(v1.size == v2.size, s"Vector dimensions do not match: Dim(v1)=${v1.size} and" +
      s" Dim(v2)=${v2.size}.")
    val v1Copy = v1.copy
    axpy(-1, v2, v1Copy)
    Vectors.norm(v1Copy, exponent)
  }

  def distance(v1: Vector, v2: Vector, exponent: Double): Double = {
    this.exponent = exponent
    distance(v1, v2)
  }

}
