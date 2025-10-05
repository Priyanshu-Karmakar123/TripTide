from langchain.prompts import PromptTemplate

PLANNER_INSTRUCTION_OG = """You are an efficient travel planner agent. You job is to plan the travel itenary and generate the travel itenary based on the reference information and the following constraints. You need to only given a travel itinerary in JSON format, along with user details and a disruption information that affects the travel plan. The disruption has a severity level — step, day, or plan — indicating how much of the itinerary is impacted. The mitigation depends on the "Disruption Tolerance" level- Flexiventurer or Planbound.
If the traveler is identified as “Planbound”, the scope of revision must strictly correspond to the disruption_severity. Specifically, for step-level disruptions, only the affected event should be modified; for day-level disruptions, modifications must be limited to the POIs scheduled for that particular day; and for plan-level disruptions, broader itinerary changes are permitted.
In contrast, for “Flexiventurer” travelers, there is no constraint linking the revision scope to the disruption severity.
Your task is to update the travel itinerary to accommodate the disruption with necessary changes, using the reference information provided to guide your modifications.
Return the complete revised travel plan in the exact same JSON format as the original. 
You must acknowledge the disruption first and then proceed with appropriate revisions based on disruption severity and traveler's disruption tolerance.
Output only the revised travel plan in strict JSON format.

*** Remember that you do not have to include the annotation plan or any explanation or the Reference Info in the output.
Once the entire travel itenary along with POI list of Day 5 ended, do not add any additional text.
Do not add the example or the PLANNER_INSTRUCTION_OG prompt in the outputs.



Given information: {text}
Query: {query}
reference_info1: {reference_info1}

###Output ### (strict JSON only, no explanations):Return only the revised itinerary JSON. Do not repeat the instruction or example."""

PLANNER_INSTRUCTION_DISRUPTION = """You are given a travel itinerary in JSON format, along with user details and a disruption information that affects the travel plan. The disruption has a severity level — step, day, or plan — indicating how much of the itinerary is impacted. The mitigation depends on the "Disruption Tolerance" level- Flexiventurer or Planbound.
If the traveler is identified as Planbound, the scope of revision must strictly correspond to the disruption_severity. Specifically, for step-level disruptions, only the affected event should be modified; for day-level disruptions, modifications must be limited to the POIs scheduled for that particular day; and for plan-level disruptions, broader itinerary changes are permitted.
In contrast, for Flexiventurer travelers, there is no constraint linking the revision scope to the disruption severity.
Your task is to update the travel itinerary to accommodate the disruption with necessary changes, using the reference information provided to guide your modifications.
Return the complete revised travel plan in the exact same JSON format as the original. 
You must acknowledge the disruption first and then proceed with appropriate revisions based on disruption severity and traveler's disruption tolerance.
 Output only the revised travel plan in strict JSON format.
 
*** Remember that you do not have to include the annotation plan or any explanation or the Reference Info in the output.
Once the entire travel itenary along with POI list of Day 7 ended, do not add any additional text.

",   

 
Travel Itinerary and User Details: {text}
Disruption Information: {query}
Reference Information1: {reference_info1}




Output the complete travel plan with acknowledgement and the modifications in the exact same JSON template as the original.
Output (Updated Travel Plan in JSON format): """

planner_agent_prompt_direct_og = PromptTemplate(
                        input_variables=["text","query","reference_info1"],
                        template = PLANNER_INSTRUCTION_DISRUPTION
                        )

# planner_agent_prompt_direct_param = PromptTemplate(
#                         input_variables=["text","query","persona"],
#                         template = PLANNER_INSTRUCTION_PARAMETER_INFO,
#                         )

# cot_planner_agent_prompt = PromptTemplate(
#                         input_variables=["text","query"],
#                         template = COT_PLANNER_INSTRUCTION,
#                         )

# react_planner_agent_prompt = PromptTemplate(
#                         input_variables=["text","query", "scratchpad"],
#                         template = REACT_PLANNER_INSTRUCTION,
#                         )

# reflect_prompt = PromptTemplate(
#                         input_variables=["text", "query", "scratchpad"],
#                         template = REFLECT_INSTRUCTION,
#                         )

# react_reflect_planner_agent_prompt = PromptTemplate(
#                         input_variables=["text", "query", "reflections", "scratchpad"],
#                         template = REACT_REFLECT_PLANNER_INSTRUCTION,
                        # )