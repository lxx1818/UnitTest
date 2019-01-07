package org.apache.spark.ml.extension.clustering.distance

import org.apache.spark.ml.linalg.BLAS._
import org.apache.spark.ml.linalg.{Vector, Vectors}

/**
  * This class implements a "Chebyshev distance" metric by finding the maximum difference
  * between each coordinate. Also 'chessboard distance' due to the moves a king can make.
  */
class ChebyshevDistanceMeasure extends DistanceMeasure {

  /**
    * @return the distance between two points.
    */
  override def distance(v1: Vector, v2: Vector): Double = {
    require(v1.size == v2.size, s"Vector dimensions do not match: Dim(v1)=${v1.size} and" +
      s" Dim(v2)=${v2.size}.")
    val v1Copy = v1.copy
    axpy(-1, v2, v1Copy)
    Vectors.norm(v1Copy, Double.PositiveInfinity)
  }

}
