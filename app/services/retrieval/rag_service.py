"""
RAG Service
Complete Retrieval-Augmented Generation pipeline
Combines retrieval with LLM generation
"""

import logging
from typing import Optional, Dict, Any, List
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class RAGConfig:
    """Configuration for RAG pipeline"""
    # Retrieval
    top_k: int = 3
    min_score: float = 0.6
    enable_reranking: bool = True
    
    # Generation
    temperature: float = 0.3
    max_tokens: int = 150
    
    # Prompt
    include_sources: bool = True
    language: str = "français"


class RAGService:
    """
    Complete RAG service combining retrieval and generation
    """
    
    def __init__(
        self,
        retrieval_service,
        lm_studio_service,
        config: Optional[RAGConfig] = None
    ):
        """
        Initialize RAG service
        
        Args:
            retrieval_service: Retrieval service instance
            lm_studio_service: LM Studio service instance
            config: RAG configuration
        """
        self.retrieval_service = retrieval_service
        self.lm_studio_service = lm_studio_service
        self.config = config or RAGConfig()
        
        logger.info("✅ RAG Service initialized")
    
    def ask(
        self,
        question: str,
        conversation_history: Optional[List[Dict[str, str]]] = None
    ) -> Dict[str, Any]:
        """
        Ask a question and get an answer using RAG
        
        Args:
            question: User's question
            conversation_history: Optional chat history
            
        Returns:
            Dictionary with answer, sources, and metadata
        """
        logger.info(f"💬 RAG Question: '{question[:100]}...'")
        
        # ================================================================
        # STEP 1: Retrieve relevant context
        # ================================================================
        logger.info("📚 Step 1: Retrieving relevant context...")
        
        retrieval_result = self.retrieval_service.retrieve(
            query=question,
            top_k=self.config.top_k,
            min_score=self.config.min_score
        )
        
        if not retrieval_result.chunks:
            logger.warning("⚠️  No relevant context found")
            return {
                "success": False,
                "answer": "Désolé, je n'ai trouvé aucune information pertinente dans mes documents pour répondre à cette question.",
                "sources": [],
                "metadata": {
                    "retrieval_time": retrieval_result.retrieval_time,
                    "chunks_found": 0
                }
            }
        
        logger.info(f"✅ Retrieved {len(retrieval_result.chunks)} chunks")
        
        # ================================================================
        # STEP 2: Construct prompt
        # ================================================================
        logger.info("📝 Step 2: Constructing prompt...")
        
        prompt = self._construct_prompt(
            question=question,
            context=retrieval_result.context,
            conversation_history=conversation_history
        )
        
        # ================================================================
        # STEP 3: Generate answer with LLM
        # ================================================================
        logger.info("🤖 Step 3: Generating answer with LLM...")
        
        answer = self.lm_studio_service.generate(
            prompt=prompt,
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens
        )
        if answer:
            # ✅ Post-process pour rendre plus concis
            answer = self._post_process_answer(answer, question)
        
        if not answer:
            logger.error("❌ Failed to generate answer")
            return {
                "success": False,
                "answer": "Désolé, une erreur s'est produite lors de la génération de la réponse.",
                "sources": [],
                "metadata": {}
            }
        
        logger.info("✅ Answer generated successfully")
        
        # ================================================================
        # STEP 4: Format response
        # ================================================================
        sources = [
            {
                "source": chunk.source,
                "relevance": chunk.score,
                "text_preview": chunk.text[:200] + "..." if len(chunk.text) > 200 else chunk.text
            }
            for chunk in retrieval_result.chunks
        ]
        
        return {
            "success": True,
            "answer": answer,
            "sources": sources,
            "metadata": {
                "retrieval_time": retrieval_result.retrieval_time,
                "chunks_found": len(retrieval_result.chunks),
                "model": self.lm_studio_service.config.model,
                "temperature": self.config.temperature
            }
        }
 
    def _construct_prompt(
        self,
        question: str,
        context: str,
        conversation_history: Optional[List[Dict[str, str]]] = None
    ) -> str:
        """Construct the prompt for the LLM"""
        
        system_prompt = f"""Tu es un assistant qui répond de manière CONCISE et PRÉCISE.

    RÈGLES:
    1. Réponds en {self.config.language} de façon DIRECTE et COURTE
    2. Base-toi UNIQUEMENT sur le contexte fourni
    3. Pour une question factuelle (nom, profession, date): 1 phrase maximum
    4. Si l'info n'est pas dans le contexte: dis "Information non trouvée dans les documents"
    5. Ne dis PAS "selon le contexte" ou "d'après les documents"
    6. Va DROIT AU BUT

    EXEMPLES:
    Q: "Quel est son nom?"
    R: "Mohamed Salah Rahmouni"

    Q: "Quelle est sa profession?"
    R: "Ingénieur full-stack"

    Q: "Où travaille-t-il?"
    R: "Information non trouvée dans les documents"
    """
        
        # Add conversation history
        history_text = ""
        if conversation_history and len(conversation_history) > 0:
            history_text = "\n## Conversation précédente:\n"
            for msg in conversation_history[-2:]:  # Only last 2 messages
                role = "User" if msg.get('role') == 'user' else "Assistant"
                content = msg.get('content', '')
                history_text += f"{role}: {content}\n"
            history_text += "\n"
        
        # Full prompt
        prompt = f"""{system_prompt}

    ## Documents disponibles:
    {context}

    {history_text}## Question de l'utilisateur:
    {question}

    ## Ta réponse (COURTE et DIRECTE):
    """
        
        return prompt
    
    def _post_process_answer(self, answer: str, question: str) -> str:
        """Post-process answer to make it more concise"""
        
        # Remove common verbose introductions
        verbose_starts = [
            "D'après le contexte fourni,",
            "Selon les informations disponibles,",
            "En me basant sur le contexte,",
            "Cette information peut être déduite",
        ]
        
        for start in verbose_starts:
            if answer.startswith(start):
                answer = answer[len(start):].strip()
                # Capitalize first letter
                if answer:
                    answer = answer[0].upper() + answer[1:]
        
        # For "what/who" questions, try to extract the direct answer
        if any(word in question.lower() for word in ['quel', 'quelle', 'qui', 'quoi', 'c\'est quoi']):
            # Remove everything after the first sentence if it's too long
            sentences = answer.split('.')
            if len(sentences) > 1 and len(sentences[0]) > 50:
                answer = sentences[0] + '.'
        
        return answer.strip()
    
    def ask_stream(
        self,
        question: str,
        conversation_history: Optional[List[Dict[str, str]]] = None
    ):
        """
        Ask a question with streaming response
        
        Args:
            question: User's question
            conversation_history: Optional chat history
            
        Yields:
            Response chunks as they're generated
        """
        logger.info(f"💬 RAG Question (streaming): '{question[:100]}...'")
        
        # Step 1: Retrieve
        retrieval_result = self.retrieval_service.retrieve(
            query=question,
            top_k=self.config.top_k,
            min_score=self.config.min_score
        )
        
        if not retrieval_result.chunks:
            yield {
                "type": "error",
                "content": "Aucune information pertinente trouvée."
            }
            return
        
        # Yield sources first
        yield {
            "type": "sources",
            "content": [
                {
                    "source": chunk.source,
                    "relevance": chunk.score
                }
                for chunk in retrieval_result.chunks
            ]
        }
        
        # Step 2: Construct prompt
        prompt = self._construct_prompt(
            question=question,
            context=retrieval_result.context,
            conversation_history=conversation_history
        )
        
        # Step 3: Stream generation
        for chunk in self.lm_studio_service.generate_stream(prompt=prompt):
            yield {
                "type": "text",
                "content": chunk
            }
        
        # Final metadata
        yield {
            "type": "metadata",
            "content": {
                "chunks_found": len(retrieval_result.chunks),
                "model": self.lm_studio_service.config.model
            }
        }