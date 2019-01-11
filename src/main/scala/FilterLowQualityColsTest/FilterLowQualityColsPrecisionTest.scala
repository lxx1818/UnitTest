package FilterLowQualityColsTest

import breeze.stats.distributions.Gaussian
import org.apache.spark.ml.preprocess.FilterLowQualityCols
import org.apache.spark.sql.SparkSession

object FilterLowQualityColsPrecisionTest {

  val featureName = "feature"

  def main(args: Array[String]): Unit = {
    print("=========开始=========\n")
    test()
    print("=========结束=========\n")
  }

    def test(): Unit = {
      val spark = SparkSession.builder().master("local[*]").appName("FilterLowQualityColsPressureTest").getOrCreate()

      val flqcDF = spark.read.format("csv")
        .option("header", "true")
        .option("inferSchema", true.toString)
        .load("/home/lxx/data/FilterLowQualityColsPrecisionTest.csv")

      val dlqcModel = new FilterLowQualityCols()
        .setMissingRatio(0.92)
        .setIDness(1.0)
        .setStability(0.96)
        .setInputCols(flqcDF.schema.names)
        .fit(flqcDF)
      val resultDF = dlqcModel.transform(flqcDF)

      val tt = resultDF.schema.map(_.metadata)
      print("ok")
    }

  def testRDD(): Unit = {
    val spark = SparkSession.builder().master("local[4]").appName("FilterLowQualityColsPressureTest").getOrCreate()

    val sampler = Gaussian(0, 1)
//
//    spark.sparkContext
//      .parallelize(0 to 1000 >> 2, 4).saveAsTextFile("123.csv")
//      .repartition(4)
//      .map { i =>
//        val samples = sampler.sample(5).mkString(",")
//        val label = scala.util.Random.nextInt(2).toString
//        samples + "," + label
//      }


    val rdd = spark.sparkContext.textFile("123.csv")


    print("ok")
  }



}
