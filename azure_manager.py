from azureml.core import Workspace, Model
from azureml.core.authentication import InteractiveLoginAuthentication
import os


class AzureManager:
    def __init__(self, subscription_id, resource_group, workspace_name):
        self.sub_id = subscription_id
        self.rg = resource_group
        self.ws_name = workspace_name
        self.workspace = None

    def connect(self):
        """Attempts to connect to an existing Azure ML Workspace."""
        try:
            # In a real app, we might use Service Principal, but Interactive is easier for local dev
            self.workspace = Workspace.get(
                name=self.ws_name,
                subscription_id=self.sub_id,
                resource_group=self.rg
            )  # Workspace retrieval
            return True, f"✅ Connected to {self.ws_name}"
        except Exception as e:
            return False, f"❌ Connection Failed: {str(e)}"

    def register_model(self, model_path, model_name):
        """Registers a trained PyTorch model to the Azure Registry."""
        if not self.workspace:
            return "❌ Not Connected to Workspace"

        try:
            model = Model.register(
                workspace=self.workspace,
                model_path=model_path,
                model_name=model_name
            )  # Model registration
            return f"🚀 Model Registered: {model.name} (v{model.version})"
        except Exception as e:
            return f"❌ Registration Failed: {str(e)}"
