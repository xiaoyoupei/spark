package com.znv.spark.mllib.image

import org.apache.spark.ml.linalg
import org.apache.spark.ml.linalg.{SparseVector, Vectors}
import org.apache.spark.sql.{DataFrame, SaveMode, SparkSession}

object Demo05ImageRead {
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


    //2、读取图片文件
    val imageData: DataFrame = spark
      .read
      .format("image")
      .load("data/train")
      .repartition(64) //重分区


    /**
      * root
      * |-- image: struct (nullable = true)
      * |    |-- origin: string (nullable = true) 文件名
      * |    |-- height: integer (nullable = true) 高度
      * |    |-- width: integer (nullable = true) 宽度
      * |    |-- nChannels: integer (nullable = true)
      * |    |-- mode: integer (nullable = true)
      * |    |-- data: binary (nullable = true) 数据
      */
    //imageData.printSchema() //查看结构


    //3、提取文件的名称以及文件的数据
    val data: DataFrame = imageData.select($"image.origin" as "name", $"image.data" as "data")


    //4、数据预处理
    val nameandfeatures: DataFrame = data
      .as[(String, Array[Byte])] //DataFrame不能map，DataSet才可map
      .map(kv => {

      val name: String = kv._1.split("/").last //文件名称

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

      val sparse: linalg.Vector = Vectors.dense(newdata) //考虑转换为稀疏向量，节省存储空间。不可行，后面会导致长度不一

      (name, sparse)
    })
      .toDF("name", "features") //重新转为DataFrame


    //5、读取图片标签
    val labelData: DataFrame = spark
      .read
      .format("csv")
      .option("sep", " ")
      .schema("name String,label Double") //名称必须设为label
      .load("data/train.txt")


    //6、特征数据和标签数据进行关联
    val resultData: DataFrame = nameandfeatures
      .join(labelData.hint("broadcast"), List("name"), "inner") //$"name"===$"name"会报错，无法识别是哪一个name，所以用List("name")
      .select("label", "features")

    //7、保存为svm
    resultData
      .write
      .mode(SaveMode.Overwrite)
      .format("libsvm")
      .save("data/images")


  }
}
