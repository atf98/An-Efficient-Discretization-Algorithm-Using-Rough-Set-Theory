import pandas as pd

class RST:
    def __init__(
        self,
        data: pd.DataFrame = None,
        continuous_columns: list = [],
        decision_column_name: str = 'class',
    ):

        
        # Primary variable of RST class
        self.data = data
        self.data_indiscernibility = {}
        self.reduct_combination = []
        self.continuous_columns = continuous_columns
        self.decision_column_name = decision_column_name
        self.lower_approx = {}
        self.upper_approx = {}
        self.boundary_region = {}
        self.outside_region = {}
        self.U = self._list_to_set_conversion(self.data.index.values.tolist())

        # Config Variable
        self.target_sets = self.get_indiscernibility([decision_column_name])
        self.target_unique = self.data[decision_column_name].unique()
        self.target_sets_dict = {self.target_unique[idx]: ele for idx, ele in enumerate(self.target_sets[decision_column_name])}

    def get_indiscernibility(self, combination: list = []):

        def indices(lst, item):
            return [i for i, x in enumerate(lst) if x == item]

        if self.decision_column_name not in combination:
            combination.append(self.decision_column_name)
            
        selected_columns = self.data[combination].to_dict('list')
        data = list(zip(*selected_columns.values()))

        self.data_indiscernibility.update({
            self._create_names(combination): list(indices(data, x) for x in set(data) if data.count(x) > 1)
        })
        return self.data_indiscernibility

    def set_main_variable(self, combination: list = []):
        _c_n = self._create_names(combination)  # combination name
        combination_upper = {}
        combination_lower = {}

        for target, set in self.target_sets_dict.items():
            if _c_n not in self.data_indiscernibility:
                self.get_indiscernibility(combination)

            c = combination.copy()
            c.append(str(target))
            __c_n = self._create_names(c)

            self.lower_approx.update({__c_n: []})
            self.upper_approx.update({__c_n: []})
            self.boundary_region.update({__c_n: []})
            self.outside_region.update({__c_n: []})

            for IND in self.data_indiscernibility[_c_n]:
                if all(True if x in set else False for x in IND):
                    self.lower_approx[__c_n].extend(IND)

                if any(True if x in set else False for x in IND):
                    self.upper_approx[__c_n].extend(IND)

            combination_lower.update({__c_n: self.lower_approx[__c_n]})
            combination_upper.update({__c_n: self.upper_approx[__c_n]})

            upper = self._list_to_set_conversion(self.upper_approx[__c_n])
            lower = self._list_to_set_conversion(self.lower_approx[__c_n])
            

            self.boundary_region[__c_n].extend(
                upper - lower
            )

            self.outside_region[__c_n].extend(
                self.U - upper
            )

        return [
            combination_upper,
            combination_lower,
            self.boundary_region[__c_n],
            self.outside_region[__c_n]
        ]

    def get_dependency(self, combination: list = []):
        _, lower, _, _ = self.set_main_variable(combination)
        divider = sum([len(v) for v in lower.values()])
        dependency = divider / len(self.U)
        return [
            lower,
            dependency
        ]

    def _create_names(self, e) -> str:
        return '_'.join(e)

    def _list_to_set_conversion(self, l):
        se = set()
        for x in l:
            se.add(x)
        return se
