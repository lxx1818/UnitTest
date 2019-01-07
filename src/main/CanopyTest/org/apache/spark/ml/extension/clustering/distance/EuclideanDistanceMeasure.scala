package org.apache.spark.ml.extension.clustering.distance

import org.apache.spark.ml.linalg.Vector

/**
  * This class implements a Euclidean distance metric by summing the square root of the squared
  * differences between each coordinate.
  *
  * If you don't care about the true distance and only need the values for comparison, then the
  * base class, SquaredEuclideanDistanceMeasure, will be faster since it doesn't do the actual
  * square root of the squared differences.
  */
class EuclideanDistanceMeasure extends SquaredEuclideanDistanceMeasure {

  /**
    * @return the distance between two points.
    */
  override def distance(v1: Vector, v2: Vector): Double = {
    Math.sqrt(super.distance(v1, v2))
  }

  /**
    * If you have special skills, use these skills to calculate the distance.
    *
    * @return the distance between two points.
    */
  override def fastDistance(v1: VectorWithNorm, v2: VectorWithNorm): Double = {
    Math.sqrt(super.fastDistance(v1, v2))
  }

  /**
    * @return the index of the closest center to the given point.
    */
  override def findClosest(
      centers: TraversableOnce[VectorWithNorm],
      point: VectorWithNorm): (Int, Double) = {
    val (bestIndex, bestDistance) = super.findClosest(centers, point)
    (bestIndex, Math.sqrt(bestDistance))
  }

}
