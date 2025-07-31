import React, { useState, useRef } from 'react';
import { Upload, FileText, Database, CheckCircle, AlertCircle, Loader2 } from 'lucide-react';

const RAGInterface = () => {
  const [brdFile, setBrdFile] = useState(null);
  const [storedProcFile, setStoredProcFile] = useState(null);
  const [extractedRules, setExtractedRules] = useState(null);
  const [isProcessing, setIsProcessing] = useState(false);
  const [error, setError] = useState(null);
  const [uploadProgress, setUploadProgress] = useState({ brd: 0, sp: 0 });
  
  const brdInputRef = useRef(null);
  const spInputRef = useRef(null);

  const handleFileUpload = (file, type) => {
    if (type === 'brd') {
      setBrdFile(file);
      setUploadProgress(prev => ({ ...prev, brd: 100 }));
    } else {
      setStoredProcFile(file);
      setUploadProgress(prev => ({ ...prev, sp: 100 }));
    }
    setError(null);
  };

  const processDocuments = async () => {
    if (!brdFile || !storedProcFile) {
      setError('Please upload both BRD document and stored procedure file');
      return;
    }

    setIsProcessing(true);
    setError(null);

    try {
      const formData = new FormData();
      formData.append('brd_document', brdFile);
      formData.append('stored_procedure', storedProcFile);

      const response = await fetch('http://localhost:8000/api/extract-rules', {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const result = await response.json();
      setExtractedRules(result);
    } catch (err) {
      setError(`Processing failed: ${err.message}`);
    } finally {
      setIsProcessing(false);
    }
  };

  const downloadRules = () => {
    if (!extractedRules) return;
    
    const dataStr = JSON.stringify(extractedRules, null, 2);
    const dataBlob = new Blob([dataStr], { type: 'application/json' });
    const url = URL.createObjectURL(dataBlob);
    const link = document.createElement('a');
    link.href = url;
    link.download = 'extracted_rules.json';
    link.click();
    URL.revokeObjectURL(url);
  };

  const FileUploadZone = ({ file, onFileSelect, type, accept, icon: Icon, label }) => (
    <div 
      className={`border-2 border-dashed rounded-lg p-6 text-center cursor-pointer transition-colors ${
        file ? 'border-green-400 bg-green-50' : 'border-gray-300 hover:border-blue-400 hover:bg-blue-50'
      }`}
      onClick={() => type === 'brd' ? brdInputRef.current?.click() : spInputRef.current?.click()}
    >
      <input
        ref={type === 'brd' ? brdInputRef : spInputRef}
        type="file"
        accept={accept}
        onChange={(e) => e.target.files?.[0] && onFileSelect(e.target.files[0], type)}
        className="hidden"
      />
      <Icon className={`mx-auto h-12 w-12 mb-4 ${file ? 'text-green-500' : 'text-gray-400'}`} />
      <p className="text-lg font-medium mb-2">{label}</p>
      {file ? (
        <div className="text-green-600">
          <CheckCircle className="inline w-5 h-5 mr-2" />
          {file.name}
        </div>
      ) : (
        <p className="text-gray-500">Click to upload or drag and drop</p>
      )}
    </div>
  );

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 to-indigo-100 p-6">
      <div className="max-w-6xl mx-auto">
        <div className="text-center mb-8">
          <h1 className="text-4xl font-bold text-gray-900 mb-4">
            RAG Document Processing System
          </h1>
          <p className="text-xl text-gray-600">
            Upload BRD documents and stored procedures to extract business rules automatically
          </p>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-2 gap-8 mb-8">
          <FileUploadZone
            file={brdFile}
            onFileSelect={handleFileUpload}
            type="brd"
            accept=".pdf"
            icon={FileText}
            label="Business Requirements Document (PDF)"
          />
          
          <FileUploadZone
            file={storedProcFile}
            onFileSelect={handleFileUpload}
            type="sp"
            accept=".sql,.txt"
            icon={Database}
            label="Stored Procedure (SQL/TXT)"
          />
        </div>

        {error && (
          <div className="mb-6 p-4 bg-red-50 border border-red-200 rounded-lg flex items-center">
            <AlertCircle className="h-5 w-5 text-red-500 mr-3" />
            <span className="text-red-700">{error}</span>
          </div>
        )}

        <div className="text-center mb-8">
          <button
            onClick={processDocuments}
            disabled={!brdFile || !storedProcFile || isProcessing}
            className={`px-8 py-3 rounded-lg font-semibold text-white transition-all ${
              !brdFile || !storedProcFile || isProcessing
                ? 'bg-gray-400 cursor-not-allowed'
                : 'bg-blue-600 hover:bg-blue-700 transform hover:scale-105'
            }`}
          >
            {isProcessing ? (
              <>
                <Loader2 className="inline w-5 h-5 mr-2 animate-spin" />
                Processing Documents...
              </>
            ) : (
              'Extract Business Rules'
            )}
          </button>
        </div>

        {extractedRules && (
          <div className="bg-white rounded-lg shadow-lg p-6">
            <div className="flex justify-between items-center mb-6">
              <h2 className="text-2xl font-bold text-gray-900">Extracted Rules</h2>
              <button
                onClick={downloadRules}
                className="px-4 py-2 bg-green-600 text-white rounded-lg hover:bg-green-700 transition-colors"
              >
                Download JSON
              </button>
            </div>
            
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6 mb-6">
              <div className="bg-blue-50 p-4 rounded-lg">
                <h3 className="font-semibold text-blue-900 mb-2">Business Rules</h3>
                <p className="text-2xl font-bold text-blue-600">
                  {extractedRules.business_rules?.length || 0}
                </p>
              </div>
              <div className="bg-green-50 p-4 rounded-lg">
                <h3 className="font-semibold text-green-900 mb-2">Data Validations</h3>
                <p className="text-2xl font-bold text-green-600">
                  {extractedRules.data_validations?.length || 0}
                </p>
              </div>
              <div className="bg-purple-50 p-4 rounded-lg">
                <h3 className="font-semibold text-purple-900 mb-2">Processing Rules</h3>
                <p className="text-2xl font-bold text-purple-600">
                  {extractedRules.processing_rules?.length || 0}
                </p>
              </div>
            </div>

            <div className="bg-gray-50 rounded-lg p-4 max-h-96 overflow-y-auto">
              <pre className="text-sm text-gray-800">
                {JSON.stringify(extractedRules, null, 2)}
              </pre>
            </div>
          </div>
        )}

        <div className="mt-12 bg-white rounded-lg shadow-lg p-6">
          <h2 className="text-2xl font-bold text-gray-900 mb-4">How it works</h2>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
            <div className="text-center">
              <div className="bg-blue-100 rounded-full w-16 h-16 flex items-center justify-center mx-auto mb-4">
                <Upload className="h-8 w-8 text-blue-600" />
              </div>
              <h3 className="font-semibold mb-2">1. Upload Documents</h3>
              <p className="text-gray-600">Upload your BRD PDF and stored procedure files</p>
            </div>
            <div className="text-center">
              <div className="bg-green-100 rounded-full w-16 h-16 flex items-center justify-center mx-auto mb-4">
                <Database className="h-8 w-8 text-green-600" />
              </div>
              <h3 className="font-semibold mb-2">2. AI Processing</h3>
              <p className="text-gray-600">RAG system analyzes and extracts business rules</p>
            </div>
            <div className="text-center">
              <div className="bg-purple-100 rounded-full w-16 h-16 flex items-center justify-center mx-auto mb-4">
                <CheckCircle className="h-8 w-8 text-purple-600" />
              </div>
              <h3 className="font-semibold mb-2">3. Get Results</h3>
              <p className="text-gray-600">Download structured JSON with extracted rules</p>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default RAGInterface;