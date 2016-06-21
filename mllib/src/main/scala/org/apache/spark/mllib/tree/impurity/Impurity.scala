/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package org.apache.spark.mllib.tree.impurity

import org.apache.spark.annotation.{DeveloperApi, Experimental, Since}

/**
 * :: Experimental ::
 * Trait for calculating information gain.
 * This trait is used for
 *  (a) setting the impurity parameter in [[org.apache.spark.mllib.tree.configuration.Strategy]]
 *  (b) calculating impurity values from sufficient statistics.
 */
@Since("1.0.0")
@Experimental
trait Impurity extends Serializable {

  /**
   * :: DeveloperApi ::
   * information calculation for multiclass classification
   * @param counts Array[Double] with counts for each label
   * @param totalCount sum of counts for all labels
   * @return information value, or 0 if totalCount = 0
   */
  @Since("1.1.0")
  @DeveloperApi
  def calculate(counts: Array[Double], totalCount: Double): Double

  /**
   * :: DeveloperApi ::
   * information calculation for regression
   * @param count number of instances
   * @param sum sum of labels
   * @param sumSquares summation of squares of the labels
   * @return information value, or 0 if count = 0
   */
  @Since("1.0.0")
  @DeveloperApi
  def calculate(count: Double, sum: Double, sumSquares: Double): Double

  @DeveloperApi
  def calculate(stats: Array[Double], start: Int, statsSize: Int): Double
}

/**
 * Interface for updating views of a vector of sufficient statistics,
 * in order to compute impurity from a sample.
 * Note: Instances of this class do not hold the data; they operate on views of the data.
 * @param statsSize  Length of the vector of sufficient statistics for one bin.
 */
private[spark] abstract class ImpurityAggregator(val statsSize: Int) extends Serializable {

  /**
   * Merge the stats from one bin into another.
   * @param allStats  Flat stats array, with stats for this (node, feature, bin) contiguous.
   * @param offset    Start index of stats for (node, feature, bin) which is modified by the merge.
   * @param otherOffset  Start index of stats for (node, feature, other bin) which is not modified.
   */
  def merge(allStats: Array[Double], offset: Int, otherOffset: Int): Unit = {
    var i = 0
    while (i < statsSize) {
      allStats(offset + i) += allStats(otherOffset + i)
      i += 1
    }
  }

  /**
   * Update stats for one (node, feature, bin) with the given label.
   * @param allStats  Flat stats array, with stats for this (node, feature, bin) contiguous.
   * @param offset    Start index of stats for this (node, feature, bin).
   */
  def update(allStats: Array[Double], offset: Int, label: Double, instanceWeight: Double): Unit

  /**
   * Get an [[ImpurityCalculator]] for a (node, feature, bin).
   * @param allStats  Flat stats array, with stats for this (node, feature, bin) contiguous.
   * @param offset    Start index of stats for this (node, feature, bin).
   */
  def getCalculator(allStats: Array[Double], offset: Int): ImpurityCalculator
}

/**
 * Stores statistics for one (node, feature, bin) for calculating impurity.
 * Unlike [[ImpurityAggregator]], this class stores its own data and is for a specific
 * (node, feature, bin).
 * @param stats  stats(start: start+statsSize) contains sufficient statistics
 * @param start  start index of stats.
 * @param statsSize  stats(start: start+statsSize) contains sufficient statistics.
 */
private[spark] abstract class ImpurityCalculator(
  var stats: Array[Double],
  private var start: Int,
  private var statsSize: Int) extends Serializable {

  /**
   * Make a deep copy of this [[ImpurityCalculator]].
   */
  def copy: ImpurityCalculator

  /**
   * Calculate the impurity from the stored sufficient statistics.
   */
  def calculate(): Double

  /**
   * Add the stats from another calculator into this one, modifying and returning this calculator.
   */
  def add(other: ImpurityCalculator): ImpurityCalculator = {
    require(statsSize == other.statsSize,
      s"Two ImpurityCalculator instances cannot be added with different counts sizes." +
        s"  Sizes are ${statsSize} and ${other.statsSize}.")
    val newStats = Array.fill[Double](statsSize)(0.0)
    var i = 0
    val thisStart = start
    val otherStart = other.start
    while (i < statsSize) {
      newStats(i) = other.stats(i + otherStart) + stats(i + thisStart)
      i += 1
    }
    stats = newStats
    start = 0
    this
  }

  /**
   * Subtract the stats from another calculator from this one, modifying and returning this
   * calculator.
   */
  def subtract(other: ImpurityCalculator): ImpurityCalculator = {
    require(statsSize == other.statsSize,
      s"Two ImpurityCalculator instances cannot be subtracted with different counts sizes." +
      s"  Sizes are ${statsSize} and ${other.statsSize}.")
    val newStats = Array.fill[Double](statsSize)(0.0)
    var i = 0
    val thisStart = start
    val otherStart = other.start
    while (i < statsSize) {
      newStats(i) = stats(i + otherStart) - other.stats(i + thisStart)
      i += 1
    }
    stats = newStats
    start = 0
    this
  }

  /**
   * Number of data points accounted for in the sufficient statistics.
   */
  def count: Long

  /**
   * Prediction which should be made based on the sufficient statistics.
   */
  def predict: Double

  /**
   * Probability of the label given by [[predict]], or -1 if no probability is available.
   */
  def prob(label: Double): Double = -1

  /**
   * Return the index of the largest array element.
   * Fails if the array is empty.
   */
  protected def indexOfLargestArrayElement(
      array: Array[Double], startInd: Int, endInd: Int): Int = {
    var largest = -1
    var largestValue = Double.MinValue
    var index = startInd
    while (index < endInd) {
      if (array(index) > largestValue) {
        largest = index
        largestValue = array(index)
      }
      index += 1
    }
    largest
  }

}

private[spark] object ImpurityCalculator {

  /**
   * Create an [[ImpurityCalculator]] instance of the given impurity type and with
   * the given stats.
   */
  def getCalculator(impurity: String, stats: Array[Double]): ImpurityCalculator = {
    impurity match {
      case "gini" => new GiniCalculator(stats, 0, stats.length)
      case "entropy" => new EntropyCalculator(stats, 0, stats.length)
      case "variance" => new VarianceCalculator(stats, 0, stats.length)
      case _ =>
        throw new IllegalArgumentException(
          s"ImpurityCalculator builder did not recognize impurity type: $impurity")
    }
  }
}
