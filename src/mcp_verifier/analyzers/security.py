"""Security analysis node for MCP server verification."""

from typing import List
import logging
from langchain_core.messages import HumanMessage, AIMessage
from langchain_anthropic import ChatAnthropic

from src.mcp_verifier.core.models import VerificationState, SecurityIssue

logger = logging.getLogger(__name__)

SECURITY_PROMPT = """
### Security Analysis for MCP Server Code

Analyze the provided MCP server code for potential security issues, focusing on the following aspects:

1. **Command Injection Vulnerabilities**  
   - Check if any external commands or system calls are executed using user input without proper sanitization or validation.

2. **Unsafe File Operations**  
   - Identify any file handling (read/write/delete) operations that might be vulnerable to directory traversal or allow arbitrary file access/modification.

3. **Insecure Dependencies**  
   - Review imported libraries or external dependencies for known vulnerabilities (e.g., outdated versions, unpatched libraries).

4. **Network Security Risks**  
   - Ensure that network communication (e.g., HTTP requests, sockets) uses secure protocols and is resistant to attacks like man-in-the-middle (MITM) or data leakage.

5. **Resource Abuse Potential (CPU, Memory, Disk)**  
   - Assess if the code can be exploited to consume excessive resources, leading to Denial-of-Service (DoS) or resource exhaustion attacks.

6. **Input Validation**  
   - Ensure that user input, especially from untrusted sources, is properly sanitized, validated, and encoded before processing to prevent injection attacks or unexpected behavior.

7. **Authentication and Authorization**  
   - Review mechanisms for authenticating users and authorizing actions. Ensure there are no flaws that could allow unauthorized access to sensitive resources.

8. **Secrets Handling**  
   - Verify that sensitive information (e.g., API keys, passwords, tokens) is stored and transmitted securely (e.g., no hardcoded secrets, proper encryption).

For each security issue found, provide:

- **Severity**: (High/Medium/Low) — Based on potential impact and exploitability.
- **Description**: (Detailed explanation) — Clear description of the security risk and its implications.
- **Location**: (File and line number) — Where the issue is located in the code.
- **Recommendation**: (How to fix) — Suggested mitigation steps to resolve the issue.

Code to analyze:
{code}
"""

class SecurityAnalyzer:
    """Analyzes MCP server code for security vulnerabilities."""
    
    def __init__(self):
        self.llm = ChatAnthropic(model="claude-3-sonnet-20240229")
        
    async def analyze(self, state: VerificationState) -> VerificationState:
        """
        Analyze server code for security issues.
        
        Args:
            state: Current verification state
            
        Returns:
            Updated state with security analysis results
        """
        logger.info("Starting security analysis")
        state.current_stage = "security_check"
        
        try:
            # Prepare code for analysis
            code_contents = []
            for file in state.files.values():
                code_contents.append(f"=== {file.path} ===\n{file.content}\n")
                
            code_text = "\n".join(code_contents)
            
            # Query LLM for security analysis
            messages = [
                HumanMessage(content=SECURITY_PROMPT.format(code=code_text))
            ]
            
            response = await self.llm.ainvoke(messages)
            
            # Parse security issues from response
            issues = self._parse_security_issues(response.content)
            state.security_issues = issues
            
            logger.info(f"Security analysis complete. Found {len(issues)} issues.")
            return state
            
        except Exception as e:
            logger.error(f"Security analysis failed: {str(e)}")
            raise
            
    def _parse_security_issues(self, response: str) -> List[SecurityIssue]:
        """Parse security issues from LLM response."""
        issues = []
        current_issue = {}
        
        for line in response.split('\n'):
            line = line.strip()
            if not line:
                continue
                
            if line.startswith('- Severity:'):
                # Save previous issue if exists
                if current_issue and len(current_issue) == 4:
                    try:
                        issues.append(SecurityIssue(**current_issue))
                    except Exception as e:
                        logger.warning(f"Failed to parse security issue: {str(e)}")
                current_issue = {}
                current_issue['severity'] = line.split(':')[1].strip().lower()
                
            elif line.startswith('- Description:'):
                current_issue['description'] = line.split(':')[1].strip()
                
            elif line.startswith('- Location:'):
                current_issue['location'] = line.split(':')[1].strip()
                
            elif line.startswith('- Recommendation:'):
                current_issue['recommendation'] = line.split(':')[1].strip()
                
        # Add last issue
        if current_issue and len(current_issue) == 4:
            try:
                issues.append(SecurityIssue(**current_issue))
            except Exception as e:
                logger.warning(f"Failed to parse security issue: {str(e)}")
                
        return issues