"""Laboratory reference ranges, fronted by the hospital LIS.

LAB DEVICE (deliberate, do not "fix"):
    This tool sleeps for LIS_LATENCY_SECONDS before returning. It simulates a slow
    Laboratory Information System lookup and is the *planted* latency in this workshop.
    In Part 4 you find this stall in a wxO observability trace without knowing in
    advance that it is here. Everything else in the team responds in well under a second.
"""

import time

from ibm_watsonx_orchestrate.agent_builder.tools import tool, ToolPermission
from pydantic import BaseModel

# The planted stall. Long enough to dominate a trace, short enough to keep the lab moving.
LIS_LATENCY_SECONDS = 4.5


class LabReference(BaseModel):
    test_name: str
    reference_range: str
    units: str
    value: float
    interpretation: str
    clinical_meaning: str
    source_system: str


_RANGES = {
    "wbc": (4.5, 11.0, "x10^9/L", "White cell count - infection, inflammation or marrow failure"),
    "crp": (0.0, 5.0, "mg/L", "C-reactive protein - acute phase inflammation"),
    "esr": (0.0, 20.0, "mm/hr", "Erythrocyte sedimentation rate - chronic inflammation"),
    "hemoglobin": (12.0, 17.5, "g/dL", "Oxygen carrying capacity"),
    "platelets": (150.0, 400.0, "x10^9/L", "Clotting capacity; low in sepsis and TTP"),
    "creatinine": (0.6, 1.3, "mg/dL", "Renal filtration"),
    "troponin": (0.0, 0.04, "ng/mL", "Myocardial injury"),
    "inr": (0.8, 1.2, "ratio", "Extrinsic clotting pathway; therapeutic 2.0-3.0 on warfarin"),
    "ldh": (140.0, 280.0, "U/L", "Cell turnover; raised in lymphoma and haemolysis"),
    "sodium": (135.0, 145.0, "mmol/L", "Osmolar balance; low sodium causes confusion"),
}


@tool(permission=ToolPermission.READ_ONLY)
def lab_reference(test_name: str, value: float) -> LabReference:
    """
    Interpret a laboratory result against the hospital reference range for that test.

    Args:
        test_name (str): The laboratory test, for example WBC, CRP, troponin, INR or sodium.
        value (float): The measured result to interpret.
    Returns:
        LabReference: The reference range, whether the value is low, normal or high, and what it means clinically.
    """
    # Planted latency - see the module docstring.
    time.sleep(LIS_LATENCY_SECONDS)

    key = test_name.strip().lower().replace(" ", "")
    if key not in _RANGES:
        raise ValueError(
            f"Unknown laboratory test '{test_name}'. Known tests: {', '.join(sorted(_RANGES))}"
        )

    low, high, units, meaning = _RANGES[key]
    if value < low:
        interpretation = "LOW"
    elif value > high:
        interpretation = "HIGH"
    else:
        interpretation = "NORMAL"

    return LabReference(
        test_name=key.upper(),
        reference_range=f"{low}-{high}",
        units=units,
        value=value,
        interpretation=interpretation,
        clinical_meaning=meaning,
        source_system="PPTH-LIS",
    )
