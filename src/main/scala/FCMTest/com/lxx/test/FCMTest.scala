package com.lxx.test

import org.apache.spark.ml.clustering.FuzzyCMeans
import org.apache.spark.sql.SparkSession


object FCMTest {

  def main(args: Array[String]): Unit = {

    print("=========开始=========\n")
//    testDim()
//    testK()
    testRows()
    print("=========结束=========\n")
  }

  def test(): Unit = {
    val spark = SparkSession.builder().getOrCreate()
    import spark.implicits._
    val dataFrame = spark.read.format("libsvm").load("/data/1600k.txt")
    val result = new FuzzyCMeans().setK(3).setM(20).setMaxIter(20).setTol(1e-40)
      .setFeaturesCol("features")
      .fit(dataFrame).transform(dataFrame)
  }

  def testRows(): Unit = {
    print("\n=========测试数据量=========\n")
    val spark = SparkSession.builder().getOrCreate()
    val path = "/data/"
    val name = Array("10", "40", "80", "100", "200", "400", "800", "1600")
    name.foreach { name =>
      print("\n=========" + name + "k=========\n")
      for (i <- 0 until 5) {
        val dataFrame = spark.read.format("libsvm").load(path + name + "k.txt")
        val result = new FuzzyCMeans().setK(3).setM(20).setMaxIter(20).setTol(1e-40)
          .setFeaturesCol("features").fit(dataFrame).transform(dataFrame)
      }
    }
  }

  def testDim(): Unit = {
    print("\n=========测试维度=========\n")
    val spark = SparkSession.builder().getOrCreate()
    val path = "/data2/"
    val name = Array("1", "5", "10", "50", "100", "200")
    name.foreach{ name =>
      print("\n=========" + name + "k=========\n")
      for (i <- 0 until 5) {
        val dataFrame = spark.read.format("libsvm").load(path + name + "k.txt")
        val result = new FuzzyCMeans().setK(3).setM(20).setMaxIter(20).setTol(1e-40)
          .setFeaturesCol("features")
          .fit(dataFrame).transform(dataFrame)
      }
    }
  }

  def testK(): Unit = {
    print("\n=========测试k值=========\n")
    val spark = SparkSession.builder().getOrCreate()
    val ks = Array(4, 8, 16, 32, 64, 128)
    ks.foreach { k =>
      print("\n========= " + k + " =========\n")
      for (i <- 0 until 5) {
        val dataFrame = spark.read.format("libsvm").load("/data/400k.txt")
        val result = new FuzzyCMeans().setK(k).setM(20).setMaxIter(20).setTol(1e-20)
          .setFeaturesCol("features")
          .fit(dataFrame).transform(dataFrame)
      }
    }
  }
}