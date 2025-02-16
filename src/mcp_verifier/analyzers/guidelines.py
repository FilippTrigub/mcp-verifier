"""Guidelines compliance analyzer for MCP server verification."""

from typing import List
import logging
from langchain_core.messages import HumanMessage, AIMessage
from langchain_anthropic import ChatAnthropic

from src.mcp_verifier.core.models import VerificationState, GuidelineViolation

logger = logging.getLogger(__name__)

GUIDELINES_PROMPT = """Analyze this MCP server implementation for compliance with community guidelines:

### Key Guidelines:

1. **Error Handling**
   - Ensure proper error messages are provided.
   - Validate that error status codes are correctly set (e.g., 4xx, 5xx).
   - Confirm that errors are propagated correctly throughout the code (e.g., raised exceptions, return values).
   - **Check**: Does the error handling logic cover all edge cases? Are there missing checks that could lead to unhandled exceptions?

2. **Rate Limiting**
   - Ensure request rate limits are enforced.
   - Check that resource usage (e.g., CPU, memory) is accounted for when imposing rate limits.
   - Verify that concurrent connections are properly limited.
   - **Check**: Are there places where rate-limiting logic might fail under high load or concurrent requests?

3. **Response Format**
   - Confirm that the response follows the standard MCP format (if applicable).
   - Validate content types (e.g., `application/json`, `application/xml`) are properly set.
   - Ensure that the response data is structured correctly and matches the expected schema.
   - **Check**: Are there inconsistencies in the response data structure? Are there potential edge cases where the format could break?

4. **Resource Management**
   - Ensure proper memory management (e.g., object cleanup, memory leaks).
   - Verify that file handles and network connections are closed after use.
   - Confirm that connection pooling (if applicable) is implemented correctly.
   - Check for appropriate timeout handling in network operations.
   - **Check**: Are there resource leaks, like open file handles or unused connections? Is there any potential inefficiency in resource usage?

5. **Documentation**
   - Ensure that API documentation is complete and accurate.
   - Include usage examples for key methods and endpoints.
   - Ensure that error conditions and their handling are well-documented.
   - **Check**: Is the documentation up to date with the code changes? Are there gaps in the explanation of complex logic?

### For each violation, provide:
- **Rule Violated**: (which specific guideline rule was violated)
- **Description**: (detailed explanation of the violation)
- **Impact**: (effect on server operation, stability, security, or performance)
- **Recommendation**: (suggested improvements to align with best practices)

For each violation, provide:
- Rule: (guideline rule violated)
- Description: (detailed explanation)
- Impact: (effect on server operation)

Server implementation:
{code}
"""

class GuidelinesAnalyzer:
    """Analyzes MCP server for community guidelines compliance."""
    
    def __init__(self):
        self.llm = ChatAnthropic(model="claude-3-sonnet-20240229")
        
    async def analyze(self, state: VerificationState) -> VerificationState:
        """
        Analyze server for guidelines compliance.
        
        Args:
            state: Current verification state
            
        Returns:
            Updated state with guidelines analysis results
        """
        logger.info("Starting guidelines compliance analysis")
        state.current_stage = "guideline_check"
        
        try:
            # Prepare code for analysis
            code_contents = []
            for file in state.files.values():
                code_contents.append(f"=== {file.path} ===\n{file.content}\n")
                
            code_text = "\n".join(code_contents)
            
            # Query LLM for guidelines analysis
            messages = [
                HumanMessage(content=GUIDELINES_PROMPT.format(code=code_text))
            ]
            
            response = await self.llm.ainvoke(messages)
            
            # Parse violations from response
            violations = self._parse_violations(response.content)
            state.guideline_violations = violations
            
            logger.info(f"Guidelines analysis complete. Found {len(violations)} violations.")
            return state
            
        except Exception as e:
            logger.error(f"Guidelines analysis failed: {str(e)}")
            raise
            
    def _parse_violations(self, response: str) -> List[GuidelineViolation]:
        """Parse guideline violations from LLM response."""
        violations = []
        current_violation = {}
        
        for line in response.split('\n'):
            line = line.strip()
            if not line:
                continue
                
            if line.startswith('- Rule:'):
                # Save previous violation if exists
                if current_violation and len(current_violation) == 3:
                    try:
                        violations.append(GuidelineViolation(**current_violation))
                    except Exception as e:
                        logger.warning(f"Failed to parse guideline violation: {str(e)}")
                current_violation = {}
                current_violation['rule'] = line.split(':')[1].strip()
                
            elif line.startswith('- Description:'):
                current_violation['description'] = line.split(':')[1].strip()
                
            elif line.startswith('- Impact:'):
                current_violation['impact'] = line.split(':')[1].strip()
                
        # Add last violation
        if current_violation and len(current_violation) == 3:
            try:
                violations.append(GuidelineViolation(**current_violation))
            except Exception as e:
                logger.warning(f"Failed to parse guideline violation: {str(e)}")
                
        return violations
        
    def get_severity_score(self, violations: List[GuidelineViolation]) -> float:
        """
        Calculate overall severity score of violations.
        
        Returns:
            Score between 0.0 (critical violations) and 1.0 (no violations)
        """
        if not violations:
            return 1.0
            
        # Count violations by impact
        critical = sum(1 for v in violations if 'critical' in v.impact.lower())
        major = sum(1 for v in violations if 'major' in v.impact.lower())
        minor = sum(1 for v in violations if 'minor' in v.impact.lower())
        
        # Weight violations
        score = 1.0
        score -= critical * 0.3  # Critical violations have high impact
        score -= major * 0.15    # Major violations have medium impact
        score -= minor * 0.05    # Minor violations have low impact
        
        return max(0.0, score)  # Ensure score is not negative