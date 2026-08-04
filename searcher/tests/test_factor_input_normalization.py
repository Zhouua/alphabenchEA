from ffo.utils.request_parsing import normalize_factors_from_expression_field


def test_structured_factor_list_preserves_name_and_expression():
    factors, error = normalize_factors_from_expression_field(
        {
            "expression": [
                {
                    "name": "BETA_10",
                    "expression": "Slope($close, 10) / $close",
                    "type": "origin",
                },
                {
                    "name": "SUMP_5",
                    "expression": "Sum(Greater($close-Ref($close,1),0), 5)",
                },
            ]
        }
    )

    assert error is None
    assert factors == [
        {"name": "BETA_10", "expression": "Slope($close, 10) / $close"},
        {"name": "SUMP_5", "expression": "Sum(Greater($close-Ref($close,1),0), 5)"},
    ]


def test_legacy_name_expression_mapping_still_works():
    factors, error = normalize_factors_from_expression_field(
        {"expression": [{"BETA_10": "Slope($close, 10) / $close"}]}
    )

    assert error is None
    assert factors == [
        {"name": "BETA_10", "expression": "Slope($close, 10) / $close"}
    ]
