package com.znv.spark.mllib

import org.apache.spark.mllib.linalg
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.regression.LabeledPoint

object Demo01Vector {
  def main (args: Array[String]): Unit = {

    //稠密向量
    val dense: linalg.Vector = Vectors.dense(Array(0.1, 0.0, 0.3, 0.0, 0.5))

    println(dense)

    //稀疏向量：只记录有值的位置
    val sparse: linalg.Vector = Vectors.sparse(5, Array(0, 2, 4), Array(0.1, 0.3, 0.5))
    println(sparse)

    //相互转换
    println(sparse.toDense)


    /**
      * LabaledPoint:一条样本数据
      */
    // Create a labeled point with a positive label and a dense feature vector.
    val pos = LabeledPoint(1.0, Vectors.dense(1.0, 0.0, 3.0))
    println(pos)

    // Create a labeled point with a negative label and a sparse feature vector.
    val neg = LabeledPoint(0.0, Vectors.sparse(3, Array(0, 2), Array(1.0, 3.0)))
    println(neg)
    

  }
}
