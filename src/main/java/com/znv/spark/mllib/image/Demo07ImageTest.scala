package com.znv.spark.mllib.image

import org.apache.spark.ml.classification.LogisticRegressionModel
import org.apache.spark.ml.linalg
import org.apache.spark.ml.linalg.{SparseVector, Vectors}
import org.apache.spark.sql.{DataFrame, SparkSession}

object Demo07ImageTest {
  def main (args: Array[String]): Unit = {
    //1、构建spark环境
    val spark: SparkSession = SparkSession
      .builder()
      .appName("image")
      .master("local[*]")
      .config("spark,sql.shuffle.partitions", "2")
      .getOrCreate()

    import spark.implicits._
    import org.apache.spark.sql.functions._

    //2、加载模型
    val model: LogisticRegressionModel = LogisticRegressionModel.load("data/imageModel")

    //3、读取图片
    val imageData: DataFrame = spark
      .read
      .format("image")
      .load("data/27550.jpg")

    val data: DataFrame = imageData.select($"image.origin" as "name", $"image.data" as "data")

    //4、提取特征
    val nameandfeatures: DataFrame = data
      .as[(String, Array[Byte])] //DataFrame不能map，DataSet才可map
      .map(kv => {

      val name: String = kv._1 //文件名称

      val value: Array[Byte] = kv._2 //文件数据

      val newdata: Array[Double] = value
        .map(_.toDouble) //将数据转为Double类型
        .map(p => { //将像素小于0的全部置1.0，其他的置0.0
        if (p < 0) {
          1.0
        } else {
          0.0
        }
      })

      val sparse: linalg.Vector = Vectors.dense(newdata)

      (name, sparse)
    })
      .toDF("name", "features")


    //5、得出结果
    model
      .transform(nameandfeatures)
      .show()


  }
}
