"""Description analysis for MCP server verification."""

import logging
from langchain_core.messages import HumanMessage, AIMessage
from langchain_anthropic import ChatAnthropic

from src.mcp_verifier.core.models import VerificationState

logger = logging.getLogger(__name__)

DESCRIPTION_PROMPT = """
Compare this MCP server implementation with its provided description.

### Areas of Focus:
1. **Feature Completeness**
   - Ensure that all features mentioned in the server description are implemented and functional.
   - Check if there are any missing or extra features in the implementation.
   
2. **Architectural Alignment**
   - Verify if the server architecture matches the described structure (e.g., rate-limiting logic, search and metadata retrieval, API design).
   - Check for alignment in the technologies and components used.

3. **Interface Compliance**
   - Ensure that the server’s API (endpoints, data structures, etc.) matches the described interface in the documentation.
   - Pay attention to input parameters, output format, and API methods.

4. **Functionality Accuracy**
   - Verify that the actual behavior of the server (requests, responses, and features) matches the described behavior.
   - Test if the server correctly implements arXiv's API guidelines, such as rate-limiting and response formats.

### Server Description:
{description}

### Implementation Code:
{code}

### Analysis Format:
1. **Implementation Summary**:
   - Provide a brief summary of the implementation.
   - Does it broadly match the description?

2. **Feature Comparison**:
   - List all features mentioned in the description and verify if they are present in the implementation.
   - Identify any missing or extra features.
   
3. **Discrepancies Found**:
   - Highlight any mismatches or issues, such as missing features, incorrect behavior, or architectural differences.

4. **Match Percentage**:
   - Provide a match percentage (0-100%) based on how well the implementation aligns with the description.

### Notes:
- When checking for discrepancies, pay special attention to:
  - The rate-limiting mechanism (1 request every 3 seconds).
  - The accuracy of the search and metadata retrieval process.
  - The response formats, ensuring they match the expected output (e.g., paper metadata, PDF/HTML links).
  - API method signatures and parameter structures.
"""

class DescriptionAnalyzer:
    """Analyzes match between server implementation and description."""
    
    def __init__(self):
        self.llm = ChatAnthropic(model="claude-3-sonnet-20240229")
        
    async def analyze(self, state: VerificationState) -> VerificationState:
        """
        Compare implementation with provided description.
        
        Args:
            state: Current verification state
            
        Returns:
            Updated state with description analysis results
        """
        logger.info("Starting description analysis")
        state.current_stage = "description_check"
        
        try:
            # Prepare code for analysis
            code_contents = []
            for file in state.files.values():
                code_contents.append(f"=== {file.path} ===\n{file.content}\n")
                
            code_text = "\n".join(code_contents)
            
            # Query LLM for description analysis
            messages = [
                HumanMessage(content=DESCRIPTION_PROMPT.format(
                    description=state.user_description,
                    code=code_text
                ))
            ]
            
            response = await self.llm.ainvoke(messages)
            
            # Parse match percentage
            match_score = self._extract_match_score(response.content)
            state.description_match = match_score
            
            logger.info(f"Description analysis complete. Match score: {match_score:.1%}")
            return state
            
        except Exception as e:
            logger.error(f"Description analysis failed: {str(e)}")
            raise
            
    def _extract_match_score(self, response: str) -> float:
        """Extract match percentage from LLM response."""
        try:
            # Look for percentage in response
            for line in response.split('\n'):
                if 'percentage' in line.lower() and '%' in line:
                    # Extract number before %
                    number = float(line.split('%')[0].split()[-1])
                    return number / 100.0
                    
            # Default to conservative match if no percentage found
            logger.warning("No match percentage found in response")
            return 0.5
            
        except Exception as e:
            logger.error(f"Failed to extract match score: {str(e)}")
            return 0.5
            
    def _analyze_discrepancies(self, response: str) -> list[str]:
        """Extract list of discrepancies from LLM response."""
        discrepancies = []
        in_discrepancies = False
        
        for line in response.split('\n'):
            line = line.strip()
            
            # Look for discrepancies section
            if 'discrepancies' in line.lower():
                in_discrepancies = True
                continue
                
            # End of discrepancies section
            if in_discrepancies and (not line or line.startswith('Match percentage')):
                break
                
            # Add discrepancy
            if in_discrepancies and line.startswith('-'):
                discrepancies.append(line[1:].strip())
                
        return discrepancies