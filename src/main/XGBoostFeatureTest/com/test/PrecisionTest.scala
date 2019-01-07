package com.test

import com.test.CanopyPrecisionTest.featureName
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.sql.SparkSession

object PrecisionTest {

  def main(args: Array[String]): Unit = {
    print("123")
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

    val iris = new VectorAssembler().setInputCols(Array("_c0", "_c1", "_c2", "_c3"))
      .setOutputCol(featureName).transform(irisDF).drop(Array("_c0", "_c1", "_c2", "_c3"): _*)


    val t = 0

  }

}
