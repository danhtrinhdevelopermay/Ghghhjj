import streamlit as st
import time
import json
from typing import List, Dict, Any, Optional, Callable
from datetime import datetime
import threading
from dataclasses import dataclass

@dataclass
class BatchItem:
    id: str
    input_text: str
    status: str = "pending"  # pending, processing, completed, failed
    response: Optional[str] = None
    error_message: Optional[str] = None
    processing_time: Optional[float] = None
    timestamp: Optional[str] = None

class BatchProcessor:
    def __init__(self):
        self.batch_items: List[BatchItem] = []
        self.current_batch_id: Optional[str] = None
        self.is_processing = False
        self.processed_count = 0
        self.total_count = 0
        self.progress_callback: Optional[Callable] = None
        
    def create_batch_from_text(self, text_input: str, delimiter: str = "\n") -> str:
        """Create a batch from delimited text input."""
        lines = [line.strip() for line in text_input.split(delimiter) if line.strip()]
        
        batch_id = f"batch_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.current_batch_id = batch_id
        
        self.batch_items = []
        for i, line in enumerate(lines):
            item = BatchItem(
                id=f"{batch_id}_{i+1:03d}",
                input_text=line
            )
            self.batch_items.append(item)
        
        self.total_count = len(self.batch_items)
        self.processed_count = 0
        
        return batch_id
    
    def create_batch_from_file(self, uploaded_file) -> str:
        """Create a batch from uploaded file (CSV, JSON, or text)."""
        batch_id = f"batch_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.current_batch_id = batch_id
        
        file_content = uploaded_file.read().decode('utf-8')
        filename = uploaded_file.name.lower()
        
        self.batch_items = []
        
        if filename.endswith('.json'):
            try:
                data = json.loads(file_content)
                if isinstance(data, list):
                    for i, item in enumerate(data):
                        if isinstance(item, str):
                            text = item
                        elif isinstance(item, dict) and 'text' in item:
                            text = item['text']
                        elif isinstance(item, dict) and 'input' in item:
                            text = item['input']
                        else:
                            text = str(item)
                        
                        batch_item = BatchItem(
                            id=f"{batch_id}_{i+1:03d}",
                            input_text=text
                        )
                        self.batch_items.append(batch_item)
                else:
                    raise ValueError("JSON file must contain an array of items")
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON file: {e}")
        
        elif filename.endswith('.csv'):
            import csv
            import io
            
            try:
                reader = csv.reader(io.StringIO(file_content))
                for i, row in enumerate(reader):
                    if i == 0 and any(header.lower() in ['text', 'input', 'prompt'] for header in row):
                        # Skip header row if it contains common column names
                        continue
                    
                    # Use first column as input text
                    text = row[0] if row else ""
                    if text.strip():
                        batch_item = BatchItem(
                            id=f"{batch_id}_{len(self.batch_items)+1:03d}",
                            input_text=text.strip()
                        )
                        self.batch_items.append(batch_item)
            except Exception as e:
                raise ValueError(f"Error reading CSV file: {e}")
        
        else:
            # Treat as plain text
            lines = [line.strip() for line in file_content.split('\n') if line.strip()]
            for i, line in enumerate(lines):
                batch_item = BatchItem(
                    id=f"{batch_id}_{i+1:03d}",
                    input_text=line
                )
                self.batch_items.append(batch_item)
        
        self.total_count = len(self.batch_items)
        self.processed_count = 0
        
        return batch_id
    
    def get_batch_status(self) -> Dict[str, Any]:
        """Get current batch processing status."""
        if not self.batch_items:
            return {"status": "no_batch", "message": "No batch created"}
        
        completed = sum(1 for item in self.batch_items if item.status == "completed")
        failed = sum(1 for item in self.batch_items if item.status == "failed")
        processing = sum(1 for item in self.batch_items if item.status == "processing")
        pending = sum(1 for item in self.batch_items if item.status == "pending")
        
        progress = (completed + failed) / self.total_count if self.total_count > 0 else 0
        
        return {
            "batch_id": self.current_batch_id,
            "total": self.total_count,
            "completed": completed,
            "failed": failed,
            "processing": processing,
            "pending": pending,
            "progress": progress,
            "is_processing": self.is_processing
        }
    
    def process_batch(self, model_manager, generation_params: Dict[str, Any], progress_callback: Optional[Callable] = None):
        """Process all items in the batch."""
        if not self.batch_items or self.is_processing:
            return
        
        self.is_processing = True
        self.progress_callback = progress_callback
        
        # Start processing in a separate thread to avoid blocking UI
        threading.Thread(target=self._process_batch_items, args=(model_manager, generation_params), daemon=True).start()
    
    def _process_batch_items(self, model_manager, generation_params: Dict[str, Any]):
        """Internal method to process batch items."""
        try:
            for i, item in enumerate(self.batch_items):
                if item.status != "pending":
                    continue
                
                item.status = "processing"
                item.timestamp = datetime.now().isoformat()
                
                # Update progress
                if self.progress_callback:
                    self.progress_callback(i + 1, self.total_count, item)
                
                start_time = time.time()
                
                try:
                    # Generate response using the model manager
                    response_data = model_manager.generate_response(
                        item.input_text,
                        **generation_params
                    )
                    
                    response = response_data.get('text')
                    if response:
                        item.response = response
                        item.status = "completed"
                    else:
                        item.status = "failed"
                        item.error_message = "No response generated"
                
                except Exception as e:
                    item.status = "failed"
                    item.error_message = str(e)
                
                item.processing_time = time.time() - start_time
                self.processed_count += 1
                
                # Add small delay to prevent overwhelming the system
                time.sleep(0.1)
        
        finally:
            self.is_processing = False
    
    def stop_processing(self):
        """Stop batch processing."""
        self.is_processing = False
    
    def get_results(self) -> List[Dict[str, Any]]:
        """Get batch processing results."""
        results = []
        for item in self.batch_items:
            result = {
                "id": item.id,
                "input": item.input_text,
                "status": item.status,
                "response": item.response,
                "error": item.error_message,
                "processing_time": item.processing_time,
                "timestamp": item.timestamp
            }
            results.append(result)
        return results
    
    def export_results(self, format_type: str = "json") -> str:
        """Export batch results in specified format."""
        results = self.get_results()
        
        if format_type.lower() == "json":
            export_data = {
                "batch_id": self.current_batch_id,
                "export_timestamp": datetime.now().isoformat(),
                "batch_summary": self.get_batch_status(),
                "results": results
            }
            return json.dumps(export_data, indent=2, ensure_ascii=False)
        
        elif format_type.lower() == "csv":
            import csv
            import io
            
            output = io.StringIO()
            writer = csv.writer(output)
            
            # Write header
            writer.writerow(["ID", "Input", "Status", "Response", "Error", "Processing Time (s)", "Timestamp"])
            
            # Write data
            for result in results:
                writer.writerow([
                    result["id"],
                    result["input"],
                    result["status"],
                    result["response"] or "",
                    result["error"] or "",
                    f"{result['processing_time']:.2f}" if result["processing_time"] else "",
                    result["timestamp"] or ""
                ])
            
            return output.getvalue()
        
        elif format_type.lower() == "txt":
            lines = [f"Batch Results - {self.current_batch_id}", "=" * 50, ""]
            
            for result in results:
                lines.append(f"ID: {result['id']}")
                lines.append(f"Input: {result['input']}")
                lines.append(f"Status: {result['status']}")
                if result['response']:
                    lines.append(f"Response: {result['response']}")
                if result['error']:
                    lines.append(f"Error: {result['error']}")
                if result['processing_time']:
                    lines.append(f"Processing Time: {result['processing_time']:.2f}s")
                lines.append("-" * 30)
                lines.append("")
            
            return "\n".join(lines)
        
        else:
            raise ValueError(f"Unsupported export format: {format_type}")
    
    def clear_batch(self):
        """Clear current batch."""
        self.batch_items = []
        self.current_batch_id = None
        self.is_processing = False
        self.processed_count = 0
        self.total_count = 0
    
    def get_failed_items(self) -> List[BatchItem]:
        """Get all failed batch items for retry."""
        return [item for item in self.batch_items if item.status == "failed"]
    
    def retry_failed_items(self, model_manager, generation_params: Dict[str, Any]):
        """Retry processing failed items."""
        failed_items = self.get_failed_items()
        if not failed_items:
            return
        
        for item in failed_items:
            item.status = "pending"
            item.error_message = None
            item.response = None
            item.processing_time = None
        
        self.process_batch(model_manager, generation_params)