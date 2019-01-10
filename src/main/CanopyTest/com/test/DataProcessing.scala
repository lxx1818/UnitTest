//package com.test
//
//import org.apache.spark.ml.extension.clustering.Canopy
//import org.apache.spark.ml.extension.clustering.distance.{DistanceMeasure, EuclideanDistanceMeasure}
//import org.apache.spark.ml.feature.VectorAssembler
//import org.apache.spark.sql.{Row, SparkSession}
//import org.apache.spark.ml.linalg._
//
//object DataProcessing {
//
//  def main(args: Array[String]): Unit = {
//    print("=========开始=========\n")
//    val files = Array("01.csv", "02.csv", "03.csv", "04.csv", "05.csv")
//    val path1 = "/home/lxx/tmp/CanopyTestResult/first/"
//    val s = Array.fill(5)("")
//    files.indices.foreach { index =>
////      s(index) = jop(path1 + files(index))
//      s(index) = jop2()
//    }
//    // jop("/home/lxx/tmp/CanopyTestResult/second/")
//
//    s.foreach(print(_))
//
//    print("=========结束=========\n")
//  }
//
//  def jop(path: String): String = {
//
//    val featureName = "feature"
//
//    val spark = SparkSession.builder().master("local[*]").appName("CanopyTest").getOrCreate()
//    val dataFrame = spark.read.format("csv")
//      .option("header", "false")
//      .option("inferSchema", true.toString)
//      .load(path)
//
//    val irisDF = spark.read.format("csv")
//      .option("header", "false")
//      .option("inferSchema", true.toString)
//      .load("/home/lxx/data/iris.data")
//
//    val iris = new VectorAssembler().setInputCols(Array("_c0", "_c1", "_c2", "_c3"))
//      .setOutputCol(featureName).transform(irisDF).drop(Array("_c0", "_c1", "_c2", "_c3"): _*)
//      .rdd.map { case Row(label: Int, feature: Vector) => (label, feature)}.collect()
//
//    val result = new VectorAssembler().setInputCols(Array("_c0", "_c1", "_c2", "_c3"))
//      .setOutputCol(featureName).transform(dataFrame)
//      .drop(Array("_c0", "_c1", "_c2", "_c3", "_c5"): _*)
//      .rdd.map { case Row(label: Int, feature: Vector) => (label, feature)}.collect()
//
//    // 平均欧式距离
//    val error = dataFrame.select("_c5").rdd
//      .map { case Row(x: Double) => x }.reduce(_+_) / 150
//
//    val arr = Array.fill(3)(Array.fill(3)(0))
//    var count = 0
//    iris.foreach { case (label, feature) =>
//      val tmp = result.filter { case (x, y) => y.equals(feature) }
//      arr(tmp(0)._1)(label) += 1
//    }
//
//    val p = arr.map(_.max).sum / 150.0
//
//    p.toString + "," + error.toString + "\n"
//  }
//
//  def jop2(): String = {
//    val featureName = "feature"
//
//    val spark = SparkSession.builder().master("local[*]").appName("CanopyTest").getOrCreate()
//
//    val irisDF = spark.read.format("csv")
//      .option("header", "false")
//      .option("inferSchema", true.toString)
//      .load("/home/lxx/data/iris.data")
//
//    val iris = new VectorAssembler().setInputCols(Array("_c0", "_c1", "_c2", "_c3"))
//      .setOutputCol(featureName).transform(irisDF).drop(Array("_c0", "_c1", "_c2", "_c3"): _*)
//
//    val model = new Canopy()
//      .setT1(1.3)
//      .setT2(1.2)
//      .setCanopyThreshold(0.0)
//      .setDistanceMeasure(DistanceMeasure.EUCLIDEAN)
//      .setFeaturesCol(featureName)
//      .setPredictionCol("pre")
//      .fit(iris)
//
//    val centers = model.getCenters
//    val arr = Array.fill(3)(Array.fill(3)(0))
//
//    val measure = new EuclideanDistanceMeasure
//
//    val result = model.transform(iris).rdd
//      .map { case Row(label: Int, feature: Vector, pre: Int) =>
//        (label, feature, pre)
//    }.collect().map { case (label, feature, pre) =>
//      arr(label)(pre) += 1
//      (label, feature, pre, measure.findClosest(centers, feature)._2)
//    }
//
//    val error = result.map(_._4).sum / 150.0
//
//    val p = arr.map(_.max).sum / 150.0
//
//    p.toString + "," + error.toString + "\n"
//
//  }
//
//}
