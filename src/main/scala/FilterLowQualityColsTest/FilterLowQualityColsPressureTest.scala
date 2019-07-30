package FilterLowQualityColsTest

import java.io.{BufferedWriter, FileOutputStream, OutputStreamWriter, PrintWriter}

import com.common.Tools
import org.apache.spark.ml.preprocess.FilterLowQualityCols
import org.apache.spark.sql.{Dataset, SparkSession}

import scala.io.Source


object FilterLowQualityColsPressureTest {

  def main(args: Array[String]): Unit = {
//    if (args(0).equals("col")) {
//      testCols()
//    } else if (args(0).equals("row")){
//      testRows()
//    }
//    generateRows()
//    testCores(args(0))
    testNullRatio()
  }

  def generateRows(): Unit = {
    val spark = SparkSession.builder().appName("gen rows data").getOrCreate()
//    val spark = Tools.getSparkSession()
    val colsNum = 10
    val k = 1000
    val rowsNumArr = Array(2, 4, 8, 10, 20)
    val fileNames = Array.fill(rowsNumArr.length)("")
    rowsNumArr.indices.foreach { indice =>
      fileNames(indice) = Tools.generateDataSet(rowsNumArr(indice) * k *10000, colsNum, 100, spark, path = "./rows/")
    }
    val out = new PrintWriter(new BufferedWriter(new OutputStreamWriter(new FileOutputStream("./rowsFileName", true))), false)
    out.print(fileNames.mkString("\n"))
    out.close()
//    Tools.write("./rowsFileName", fileNames)
  }

  def generateCols(): Unit = {
    val spark = SparkSession.builder().appName("gen cols data").getOrCreate()
    //    val spark = Tools.getSparkSession()
    val colsNumArr = Array(10, 20, 40, 80, 200, 400, 600, 1000)
    val k = 1000
    val rowsNum = 10000 * k
    val fileNames = Array.fill(colsNumArr.length)("")
    colsNumArr.indices.foreach { indice =>
      fileNames(indice) = Tools.generateDataSet(rowsNum, colsNumArr(indice), 100, spark, path = "./cols/")
    }
    Tools.write("./colsFileName", fileNames)
  }


  def generateNullRatio(): Unit = {
    val spark = SparkSession.builder().appName("gen null data").getOrCreate()
    //    val spark = Tools.getSparkSession()
    val nullRatio = Array(0.0, 0.1, 0.2, 0.4, 0.6, 0.8)
    val k = 1000
    val rowsNum = 10000 * k
    val fileNames = Array.fill(nullRatio.length)("")
    nullRatio.indices.foreach { indice =>
      fileNames(indice) = Tools.generateDataSet(rowsNum, 100, 100, spark, path = "./nulls/", nullRatio = nullRatio(indice))
    }
    Tools.write("./nullsFileName", fileNames)
  }


  def testRows(): Unit = {
    val spark = SparkSession.builder().appName("test rows").getOrCreate()

    val source = Source.fromFile("/home/lxx/DataSet/rowResult", "UTF-8")
    val lineIterator =source.getLines
    for(file <- lineIterator) {
      val path = "/home/lxx/DataSet/rows/" + file
      val out = new PrintWriter(new BufferedWriter(new OutputStreamWriter(new FileOutputStream("./rowResult", true))), false)
      out.print(file)
      val df = Tools.readDataFrameASInteger(path, spark).repartition(100)
      df.cache()
      df.first()
      for (i <- 0 to 4) {
        val time = run(df)
        out.print("," + time)
      }
      out.println(",rows:" + df.count() + ",cols:" + df.first().length)
      df.unpersist()
      out.close
    }
  }

  def testCores(core: String): Unit = {
    val out = new PrintWriter(new BufferedWriter(new OutputStreamWriter(new FileOutputStream("./coreResult2", true))), false)
    val spark = SparkSession.builder().appName("test core").getOrCreate()
    val path = "/home/lxx/DataSet/rows/rowNum10000000_featureNum10_labelfalse_nullRatio0.0"
    val df = Tools.readDataFrameASInteger(path, spark).repartition(100)
    out.print(core)
    df.cache()
    df.first()
    for (i <- 0 to 4) {
      val time = run(df)
      out.print("," + time)
    }
    out.println(",rows:" + df.count() + ",cols:" + df.first().length)
    df.unpersist()
    spark.close()
    out.close()
  }

  def testCols(): Unit = {
    val spark = SparkSession.builder().appName("test cols").getOrCreate()

    val source = Source.fromFile("/home/lxx/DataSet/colsFileName", "UTF-8")
    val lineIterator =source.getLines
    for(file <- lineIterator) {
      val path = "/home/lxx/DataSet/cols/" + file
      val out = new PrintWriter(new BufferedWriter(new OutputStreamWriter(new FileOutputStream("./colResult", true))), false)
      out.print(file)
      val df = Tools.readDataFrameASInteger(path, spark).repartition(100)
      df.cache()
      df.first()
      for (i <- 0 to 4) {
        val time = run(df)
        out.print("," + time)
      }
      out.println(",rows:" + df.count() + ",cols:" + df.first().length)
      df.unpersist()
      out.close
    }
  }

  def testNullRatio(): Unit = {
    val spark = SparkSession.builder().appName("test null ratio").getOrCreate()

    val source = Source.fromFile("/home/lxx/DataSet/nullsFileName", "UTF-8")
    val lineIterator =source.getLines
    for(file <- lineIterator) {
      val path = "/home/lxx/DataSet/nulls/" + file
      val out = new PrintWriter(new BufferedWriter(new OutputStreamWriter(new FileOutputStream("./nullResult", true))), false)
      out.print(file)
      val df = Tools.readDataFrameASInteger(path, spark).repartition(100)
      df.cache()
      df.first()
      for (i <- 0 to 4) {
        val time = run(df)
        out.print("," + time)
      }
      out.println(",rows:" + df.count() + ",cols:" + df.first().length)
      df.unpersist()
      out.close
    }
  }


  // 运行FilterLowQualityCols，返回运行时间（秒）
  def run(dataset: Dataset[_]): Double = {
    val names = dataset.schema.names
    val iterationStartTime = System.nanoTime()
    val resultDF = new FilterLowQualityCols()
      .setMissingRatio(0.8)
      .setIDness(0.8)
      .setStability(0.8)
      .setInputCols(names)
      .fit(dataset).transform(dataset)
    (System.nanoTime() - iterationStartTime) / 1e9
  }
}
