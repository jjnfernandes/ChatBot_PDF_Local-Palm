from langchain.prompts import PromptTemplate

class SystemPrompt:
    def questionPrompt(self):
        condense_qa_template = """
            Given the following conversation and a follow up question, rephrase the follow up question
            to be a standalone question, and add an instruction in the question to answer from the contexts provided.

            Chat History:
            {chat_history}
            Follow Up Input: {question}
            Standalone question:"""
        
        CUSTOM_QUESTION_PROMPT = PromptTemplate.from_template(condense_qa_template)

        return CUSTOM_QUESTION_PROMPT