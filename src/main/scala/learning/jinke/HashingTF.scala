package learning.jinke

import org.apache.spark.mllib.feature.HashingTF
import org.apache.spark.{SparkConf, SparkContext}

/**
  * HashTF: 词频特征
  */
object HashingTF {

  def main(args: Array[String]): Unit = {
    val sc = new SparkContext(new SparkConf().setAppName("App").setMaster("local[*]"))

    val lines = sc.textFile("data/readme.txt")

    //词频特征的维数10维
    val hashingTF = new HashingTF(numFeatures = 10)

    //每一行映射为一个特征
    val lineFeature = lines.map(line=> hashingTF.transform(line.split("\\s+")))
//    val linesFeature = hashingTF.transform(lines)
    lineFeature.take(5).foreach(feature=> println(feature.toArray.mkString(" , ")))
    println(lineFeature.count)
  }
}
