"""
The pairwise discrimination rate: how often a near-native pose is ranked higher than a 
    non-near-native one.

It is Mann-Whitney's U over the two groups, scaled by the pairs they make, so `scipy.stats` computes
    it. A ranking is handed as a score a pose, higher the better, so an energy arrives negated.

One rate a complex, which is what the report means over, so a complex with many poses does not
    dominate it. A pose whose RMSD was never measured is left out.
"""

from utils import report

NEAR, OTHER, UNMEASURED = True, False, None


def test_a_ranking_is_rated_on_the_pairs_it_orders_the_right_way():
    assert report.pairwise_discriminate([3.0, 2.0, 1.0, 0.0], [NEAR, NEAR, OTHER, OTHER]) == 1.0
    assert report.pairwise_discriminate([1.0, 0.0, 3.0, 2.0], [NEAR, NEAR, OTHER, OTHER]) == 0.0
    # Three of the four pairs: the near-native pose scoring 1.5 sits under the other scoring 2.0.
    assert report.pairwise_discriminate([3.0, 1.5, 2.0, 1.0], [NEAR, NEAR, OTHER, OTHER]) == 0.75


def test_a_pair_the_ranking_scores_the_same_counts_a_half():
    assert report.pairwise_discriminate([1.0, 1.0], [NEAR, OTHER]) == 0.5
    assert report.pairwise_discriminate([2.0, 2.0, 1.0], [NEAR, OTHER, OTHER]) == 0.75


def test_a_pose_whose_rmsd_was_never_measured_makes_no_pair():
    # Left in, its 9.0 would take the near-native pose's rate to two thirds.
    assert report.pairwise_discriminate([9.0, 3.0, 2.0, 1.0], [UNMEASURED, NEAR, OTHER, OTHER]) == 1.0


def test_a_complex_with_no_pair_to_order_has_no_rate():
    assert report.pairwise_discriminate([3.0, 2.0], [NEAR, NEAR]) is None
    assert report.pairwise_discriminate([3.0, 2.0], [OTHER, OTHER]) is None
    assert report.pairwise_discriminate([3.0, 2.0], [NEAR, UNMEASURED]) is None
