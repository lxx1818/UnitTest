package org.apache.spark.ml.extension.clustering.distance

import org.apache.spark.ml.linalg.{Vector, Vectors}

/** This trait is used for objects which can determine a distance metric between two points */
trait DistanceMeasure extends Serializable {

  /**
    * @return the distance between two points.
    */
  def distance(v1: Vector, v2: Vector): Double

  /**
    * If you have special skills, use these skills to calculate the distance.
    *
    * @return the distance between two points.
    */
  def fastDistance(v1: VectorWithNorm, v2: VectorWithNorm): Double = {
    distance(v1.vector, v2.vector)
  }

  /**
    * @return the index of the closest center to the given point.
    */
  def findClosest(centers: TraversableOnce[Vector], point: Vector): (Int, Double) = {
    var bestDistance = Double.PositiveInfinity
    var bestIndex = 0
    var i = 0
    centers.foreach { center =>
      val currentDistance = distance(center, point)
      if (currentDistance < bestDistance) {
        bestDistance = currentDistance
        bestIndex = i
      }
      i += 1
    }
    (bestIndex, bestDistance)
  }

  /**
    * @return the index of the closest center to the given point.
    */
  def findClosest(
      centers: TraversableOnce[VectorWithNorm],
      point: VectorWithNorm): (Int, Double) = {
    var bestDistance = Double.PositiveInfinity
    var bestIndex = 0
    var i = 0
    centers.foreach { center =>
      val currentDistance = fastDistance(center, point)
      if (currentDistance < bestDistance) {
        bestDistance = currentDistance
        bestIndex = i
      }
      i += 1
    }
    (bestIndex, bestDistance)
  }
}

object DistanceMeasure {

  val CHEBYSHEV = "chebyshev"
  val COSINE = "cosine"
  val EUCLIDEAN = "euclidean"
  val MANHATTAN = "manhattan"
  val MINKOWSKI = "minkowski"
  val SQUAREDEUCLIDEAN = "squaredEuclidean"
  val TANIMOTO = "tanimoto"
  val supportedDistanceMeasure = Array(CHEBYSHEV, COSINE, EUCLIDEAN, MANHATTAN, MINKOWSKI,
    SQUAREDEUCLIDEAN, TANIMOTO)

  def decodeFromString(distanceMeasure: String): DistanceMeasure =
    distanceMeasure match {
      case CHEBYSHEV => new ChebyshevDistanceMeasure
      case COSINE => new CosineDistanceMeasure
      case EUCLIDEAN => new EuclideanDistanceMeasure
      case MANHATTAN => new ManhattanDistanceMeasure
      case MINKOWSKI => new MinkowskiDistanceMeasure
      case SQUAREDEUCLIDEAN => new SquaredEuclideanDistanceMeasure
      case TANIMOTO => new TanimotoDistanceMeasure
      case _ => throw new IllegalArgumentException(s"distanceMeasure must be one of: " +
        s"${supportedDistanceMeasure.mkString(", ")}. $distanceMeasure provided.")
    }

  def validateDistanceMeasure(distanceMeasure: String): Boolean = {
    supportedDistanceMeasure.exists(_.equals(distanceMeasure))
  }

}

class VectorWithNorm(val vector: Vector, val norm: Double) extends Serializable {

  def this(vector: Vector) = this(vector, Vectors.norm(vector, 2.0))

  def this(array: Array[Double]) = this(Vectors.dense(array))

  /** Converts the vector to a dense vector. */
  def toDense: VectorWithNorm = new VectorWithNorm(Vectors.dense(vector.toArray), norm)
}