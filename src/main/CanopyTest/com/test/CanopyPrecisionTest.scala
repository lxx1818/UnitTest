package com.test

import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.ml.extension.clustering.Canopy
import org.apache.spark.ml.extension.clustering.distance.DistanceMeasure
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.linalg.Vector
import org.apache.spark.sql.{Row, SparkSession}

object CanopyPrecisionTest {

  val featureName = "feature"

  def main(args: Array[String]): Unit = {
    print("=========开始=========\n")
    test11()
    print("=========结束=========\n")
  }

  def test11(): Unit = {
    val spark = SparkSession.builder().master("local[*]").appName("CanopyTest").getOrCreate()

    val irisDF = spark.read.format("csv")
      .option("header", "false")
      .option("inferSchema", true.toString)
      .load("/home/lxx/data/iris.data")

    val iris = new VectorAssembler().setInputCols(Array("_c0", "_c1", "_c2", "_c3"))
      .setOutputCol(featureName).transform(irisDF).drop(Array("_c0", "_c1", "_c2", "_c3"): _*)
      .filter { x => x.get(0) != 2}

    val result = new LogisticRegression()
      .setFeaturesCol(featureName)
      .setLabelCol("_c4")
      .fit(iris).transform(iris)

    print("ok")
  }

  def test(): Unit = {
    val spark = SparkSession.builder().master("local[*]").appName("CanopyTest").getOrCreate()
    import spark.implicits._
    val dataFrame = spark.read.format("csv")
      .option("header", "false")
      .option("inferSchema", true.toString)
      .load("/home/lxx/data/stamps.txt")
    val featuresName = Array("_c0", "_c1", "_c2", "_c3", "_c4", "_c5", "_c6", "_c7", "_c8")
    val inputDF = new VectorAssembler()
      .setInputCols(featuresName)
      .setOutputCol("features")
      .transform(dataFrame).drop(featuresName: _*)

    val canopy = new Canopy()
      .setT1(0.5)
      .setT2(0.3)
      .setCanopyThreshold(3.0)
      .setDistanceMeasure(DistanceMeasure.EUCLIDEAN)
      .setFeaturesCol("features")
      .setPredictionCol("pre")
      .setCanopiesCol("canopy")

    val result = canopy.fit(inputDF).transform(inputDF)
  }

  def testWine(): Unit ={
    val spark = SparkSession.builder().master("local[*]").appName("CanopyTest").getOrCreate()
    val wineDF = spark.read.format("csv")
      .option("header", "false")
      .option("inferSchema", true.toString)
      .load("/home/lxx/data/wine.data")

    val t = 0
  }

  def testIris(): Unit ={
    val spark = SparkSession.builder().master("local[*]").appName("CanopyTest").getOrCreate()
    val irisDF = spark.read.format("csv")
      .option("header", "false")
      .option("inferSchema", true.toString)
      .load("/home/lxx/data/iris.data")
    val t = 0

  }


  

}
