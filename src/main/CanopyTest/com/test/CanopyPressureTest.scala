//package com.test
//
//import org.apache.spark.sql.SparkSession
//
//object CanopyPressureTest {
//
//  val Ciphertext = "\nCiphertext1596876248dfsa\t"
//
//  def main(args: Array[String]): Unit = {
////    testRows()
////    testDim()
////    testK()
//    testCores()
//  }
//
//  def testRows(): Unit = {
//    print(Ciphertext + "=========测试数据量=========\n")
//    val spark = SparkSession.builder().getOrCreate()
//    val path = "/canopyData/rows/"
//    val names = Array("10", "50", "100", "200", "300", "400", "800", "1600")
////    val names = Array("10")
//    names.foreach { name =>
//      print(Ciphertext + "=========" + name + "k=========\n")
//      val dataFrame = spark.read.format("libsvm").load(path + name + "k.libsvm")
//      for (i <- 0 until 5) {
//        val iterationStartTime = System.nanoTime()
//        new Canopy().setT1(20).setT2(10)
//          .setDistanceMeasure(DistanceMeasure.EUCLIDEAN)
//          .setFeaturesCol("features")
//          .setPredictionCol("prediction")
//          .setCanopiesCol("canopy")
//          .fit(dataFrame).transform(dataFrame)
//        val iterationTimeInSeconds = (System.nanoTime() - iterationStartTime) / 1e9
//        println(Ciphertext + "time\t" + iterationTimeInSeconds)
//      }
//    }
//    spark.close()
//  }
//
//  def testDim(): Unit = {
//    print(Ciphertext + "=========测试维度=========\n")
//    val spark = SparkSession.builder().getOrCreate()
//    val path = "/canopyData/dims/"
//    val names = Array("1", "5", "10", "20", "40", "80", "160")
//    val arrT2 = Array(10.0, 23.0, 32.0, 45.0, 64.0, 90.0, 127.0)
//    names.indices.foreach{ index =>
//      val name = names(index)
//      val t2 = arrT2(index)
//      print(Ciphertext + "=========" + name + "k=========\n")
//      val dataFrame = spark.read.format("libsvm").load(path + name + "k.libsvm")
//      for (i <- 0 until 5) {
//        val iterationStartTime = System.nanoTime()
//        new Canopy().setT1(t2 * 2).setT2(t2)
//          .setDistanceMeasure(DistanceMeasure.EUCLIDEAN)
//          .setFeaturesCol("features")
//          .setPredictionCol("prediction")
//          .setCanopiesCol("canopy")
//          .fit(dataFrame).transform(dataFrame)
//        val iterationTimeInSeconds = (System.nanoTime() - iterationStartTime) / 1e9
//        println(Ciphertext + "time\t" + iterationTimeInSeconds)
//      }
//    }
//  }
//
//  def testK(): Unit = {
//    print(Ciphertext + "=========测试k值=========\n")
//    val spark = SparkSession.builder().getOrCreate()
//    val path = "/canopyData/k/"
////    val names = Array("2", "4", "8", "16", "32", "64", "96")
//    val names = Array("100", "200", "400", "800", "1600", "3200")
//    names.foreach { name =>
//      print(Ciphertext + "=========" + name + "c=========\n")
//      val dataFrame = spark.read.format("libsvm").load(path + name + "c.libsvm")
//      for (i <- 0 until 5) {
//        val iterationStartTime = System.nanoTime()
//        new Canopy().setT1(20).setT2(10)
//          .setDistanceMeasure(DistanceMeasure.EUCLIDEAN)
//          .setFeaturesCol("features")
//          .setPredictionCol("prediction")
//          .setCanopiesCol("canopy")
//          .fit(dataFrame).transform(dataFrame)
//        val iterationTimeInSeconds = (System.nanoTime() - iterationStartTime) / 1e9
//        println(Ciphertext + "time\t" + iterationTimeInSeconds)
//      }
//    }
//  }
//
//  def testCores(): Unit = {
//    print(Ciphertext + "=========测试k值=========\n")
//    val spark = SparkSession.builder().getOrCreate()
//    val path = "/canopyData/rows/1600k.libsvm"
//    val dataFrame = spark.read.format("libsvm").load(path)
//    for (i <- 0 until 5) {
//      val iterationStartTime = System.nanoTime()
//      new Canopy().setT1(20).setT2(10)
//        .setDistanceMeasure(DistanceMeasure.EUCLIDEAN)
//        .setFeaturesCol("features")
//        .setPredictionCol("prediction")
//        .setCanopiesCol("canopy")
//        .fit(dataFrame).transform(dataFrame)
//      val iterationTimeInSeconds = (System.nanoTime() - iterationStartTime) / 1e9
//      println(Ciphertext + "time\t" + iterationTimeInSeconds)
//    }
//  }
//
//}
