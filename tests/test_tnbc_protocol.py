import unittest

from train_tnbc_grouped_cv import split_cases


class TNBCGroupedProtocol(unittest.TestCase):
    def test_every_patient_is_disjoint_and_tested_once(self):
        folds = [["04", "11"], ["01", "02"], ["09", "07"], ["03", "10"], ["05", "08", "06"]]
        cases = [
            {"id": f"{patient}_1", "group": patient}
            for patient in sorted(set().union(*map(set, folds)))
        ]
        tested = []
        for fold in range(len(folds)):
            train, validation, test = split_cases(cases, folds, fold)
            train_groups = {case["group"] for case in train}
            validation_groups = {case["group"] for case in validation}
            test_groups = {case["group"] for case in test}
            self.assertFalse(train_groups & validation_groups)
            self.assertFalse(train_groups & test_groups)
            self.assertFalse(validation_groups & test_groups)
            self.assertEqual(test_groups, set(folds[fold]))
            self.assertEqual(validation_groups, set(folds[(fold + 1) % len(folds)]))
            tested.extend(case["id"] for case in test)
        self.assertCountEqual(tested, [case["id"] for case in cases])


if __name__ == "__main__":
    unittest.main()
