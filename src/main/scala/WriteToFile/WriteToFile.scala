package WriteToFile

import java.io.{BufferedWriter, FileOutputStream, OutputStreamWriter, PrintWriter}

import scala.io.Source

object WriteToFile {
  def main(args: Array[String]): Unit = {

    write()
    write()
  }

  def read(): Unit = {

    val source = Source.fromFile("/home/lxx/data/FilterLowQualityColsTestDataSet.csv", "UTF-8")
    val lineIterator =source.getLines

    for(l<-lineIterator)
      println(l)
    //或者可以对迭代器应用toArray或toBuffer方法，将行放到数组或数组缓冲当中
    val lines = source.getLines.toArray
    //将整个文件读取成一个字符串
    //val contents = source.mkString;
    source.close

  }

  def write(): Unit = {
    val path = "./testlxx"
    val out = new PrintWriter(new BufferedWriter(new OutputStreamWriter(new FileOutputStream(path, true))), false)
    for(i<-1 to 10)
      out.println(i)
    out.close
  }

}
