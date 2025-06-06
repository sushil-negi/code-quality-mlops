#!/usr/bin/env python3
"""
MLOps Module Selector

Interactive tool to help users select and configure MLOps modules
based on their requirements, budget, and use case.
"""

import json
import sys
import os
from dataclasses import dataclass
from typing import Dict, List, Optional
import argparse

@dataclass
class ModuleConfig:
    name: str
    description: str
    cost_factor: float  # Relative cost multiplier
    complexity: str     # "low", "medium", "high"
    best_for: List[str]
    pros: List[str]
    cons: List[str]

class MLOpsModuleSelector:
    def __init__(self):
        self.modules = {
            "data_pipeline": {
                "kafka": ModuleConfig(
                    name="Apache Kafka (MSK)",
                    description="Distributed streaming platform with high throughput",
                    cost_factor=1.5,
                    complexity="medium",
                    best_for=["high-throughput", "real-time", "production"],
                    pros=["High throughput", "Battle-tested", "Rich ecosystem"],
                    cons=["Higher cost", "Complex configuration", "Requires expertise"]
                ),
                "kinesis": ModuleConfig(
                    name="AWS Kinesis",
                    description="Managed streaming service with serverless scaling",
                    cost_factor=1.0,
                    complexity="low",
                    best_for=["aws-native", "serverless", "simple-setup"],
                    pros=["Fully managed", "Auto-scaling", "AWS integration"],
                    cons=["AWS vendor lock-in", "Limited customization", "Shard management"]
                ),
                "pulsar": ModuleConfig(
                    name="Apache Pulsar",
                    description="Cloud-native messaging with built-in multi-tenancy",
                    cost_factor=1.3,
                    complexity="high",
                    best_for=["multi-tenant", "geo-replication", "cloud-native"],
                    pros=["Multi-tenancy", "Geo-replication", "Schema evolution"],
                    cons=["Newer technology", "Steeper learning curve", "Less tooling"]
                )
            },
            "ml_platform": {
                "mlflow": ModuleConfig(
                    name="MLflow",
                    description="Open source ML lifecycle management",
                    cost_factor=0.5,
                    complexity="low",
                    best_for=["cost-conscious", "flexibility", "open-source"],
                    pros=["Open source", "Language agnostic", "Simple setup"],
                    cons=["Limited enterprise features", "Scaling challenges", "Manual ops"]
                ),
                "kubeflow": ModuleConfig(
                    name="Kubeflow",
                    description="ML workflows on Kubernetes",
                    cost_factor=1.0,
                    complexity="high",
                    best_for=["kubernetes", "workflows", "scalability"],
                    pros=["Kubernetes native", "Workflow orchestration", "Scalable"],
                    cons=["Complex setup", "Kubernetes expertise required", "Resource intensive"]
                ),
                "sagemaker": ModuleConfig(
                    name="AWS SageMaker",
                    description="Fully managed ML platform",
                    cost_factor=2.0,
                    complexity="low",
                    best_for=["managed-service", "enterprise", "aws-integration"],
                    pros=["Fully managed", "Enterprise features", "Auto-scaling"],
                    cons=["Expensive", "AWS lock-in", "Less flexibility"]
                )
            },
            "monitoring": {
                "prometheus": ModuleConfig(
                    name="Prometheus + Grafana",
                    description="Open source monitoring and alerting",
                    cost_factor=0.3,
                    complexity="medium",
                    best_for=["cost-conscious", "kubernetes", "customization"],
                    pros=["Open source", "Highly customizable", "Rich ecosystem"],
                    cons=["Requires setup", "Storage scaling", "Alerting complexity"]
                ),
                "datadog": ModuleConfig(
                    name="DataDog",
                    description="SaaS monitoring and APM",
                    cost_factor=2.5,
                    complexity="low",
                    best_for=["enterprise", "apm", "easy-setup"],
                    pros=["Easy setup", "Rich features", "Great UX"],
                    cons=["Expensive", "Data egress costs", "Vendor lock-in"]
                ),
                "cloudwatch": ModuleConfig(
                    name="AWS CloudWatch",
                    description="AWS native monitoring",
                    cost_factor=1.0,
                    complexity="low",
                    best_for=["aws-native", "basic-monitoring", "cost-effective"],
                    pros=["AWS integrated", "No setup", "Cost effective"],
                    cons=["AWS only", "Limited features", "Query limitations"]
                )
            },
            "serving": {
                "kubernetes": ModuleConfig(
                    name="Kubernetes (EKS)",
                    description="Container orchestration platform",
                    cost_factor=1.2,
                    complexity="high",
                    best_for=["scalability", "flexibility", "microservices"],
                    pros=["Highly scalable", "Flexible", "Industry standard"],
                    cons=["Complex", "Operational overhead", "Learning curve"]
                ),
                "ecs": ModuleConfig(
                    name="AWS ECS/Fargate",
                    description="AWS managed container service",
                    cost_factor=1.0,
                    complexity="medium",
                    best_for=["aws-native", "containers", "managed-service"],
                    pros=["AWS managed", "Less complexity", "Good integration"],
                    cons=["AWS lock-in", "Less flexible", "Limited ecosystem"]
                ),
                "lambda": ModuleConfig(
                    name="AWS Lambda",
                    description="Serverless compute",
                    cost_factor=0.5,
                    complexity="low",
                    best_for=["serverless", "event-driven", "cost-optimization"],
                    pros=["No servers", "Pay per use", "Auto-scaling"],
                    cons=["Cold starts", "Runtime limits", "Vendor lock-in"]
                )
            }
        }
        
        self.use_cases = {
            "startup": {
                "budget": "low",
                "complexity_preference": "low",
                "priorities": ["cost", "simplicity", "speed"]
            },
            "enterprise": {
                "budget": "high",
                "complexity_preference": "medium",
                "priorities": ["reliability", "security", "support"]
            },
            "research": {
                "budget": "medium",
                "complexity_preference": "high",
                "priorities": ["flexibility", "experimentation", "cutting-edge"]
            },
            "production": {
                "budget": "medium-high",
                "complexity_preference": "medium",
                "priorities": ["reliability", "scalability", "performance"]
            }
        }

    def interactive_selection(self):
        """Interactive module selection wizard"""
        print("🚀 MLOps Module Selector")
        print("=" * 50)
        
        # Get user requirements
        use_case = self._get_use_case()
        budget = self._get_budget()
        complexity_tolerance = self._get_complexity_tolerance()
        specific_requirements = self._get_specific_requirements()
        
        # Generate recommendations
        recommendations = self._generate_recommendations(
            use_case, budget, complexity_tolerance, specific_requirements
        )
        
        # Display recommendations
        self._display_recommendations(recommendations)
        
        # Generate terraform configuration
        if self._confirm_selection(recommendations):
            self._generate_terraform_config(recommendations)
    
    def _get_use_case(self) -> str:
        print("\n📋 What's your primary use case?")
        use_cases = list(self.use_cases.keys())
        for i, case in enumerate(use_cases, 1):
            description = self.use_cases[case]
            print(f"  {i}. {case.title()} - Budget: {description['budget']}, "
                  f"Complexity: {description['complexity_preference']}")
        
        while True:
            try:
                choice = int(input("\nSelect use case (1-4): ")) - 1
                if 0 <= choice < len(use_cases):
                    return use_cases[choice]
                else:
                    print("Invalid choice. Please select 1-4.")
            except ValueError:
                print("Please enter a number.")
    
    def _get_budget(self) -> str:
        print("\n💰 What's your monthly budget range?")
        budgets = {
            1: ("low", "$300-800", "Optimized for cost"),
            2: ("medium", "$800-1500", "Balanced cost/performance"),
            3: ("high", "$1500-3000", "Performance focused"),
            4: ("unlimited", "$3000+", "Enterprise grade")
        }
        
        for key, (_, range_str, desc) in budgets.items():
            print(f"  {key}. {range_str} - {desc}")
        
        while True:
            try:
                choice = int(input("\nSelect budget range (1-4): "))
                if choice in budgets:
                    return budgets[choice][0]
                else:
                    print("Invalid choice. Please select 1-4.")
            except ValueError:
                print("Please enter a number.")
    
    def _get_complexity_tolerance(self) -> str:
        print("\n🔧 How much operational complexity can you handle?")
        complexity_levels = {
            1: ("low", "Simple setup, managed services preferred"),
            2: ("medium", "Some configuration acceptable"),
            3: ("high", "Complex setups fine, prefer flexibility")
        }
        
        for key, (level, desc) in complexity_levels.items():
            print(f"  {key}. {level.title()} - {desc}")
        
        while True:
            try:
                choice = int(input("\nSelect complexity tolerance (1-3): "))
                if choice in complexity_levels:
                    return complexity_levels[choice][0]
                else:
                    print("Invalid choice. Please select 1-3.")
            except ValueError:
                print("Please enter a number.")
    
    def _get_specific_requirements(self) -> List[str]:
        print("\n🎯 Any specific requirements? (select multiple)")
        requirements = [
            "real-time-processing",
            "batch-processing",
            "auto-scaling",
            "multi-cloud",
            "on-premise",
            "compliance",
            "high-availability",
            "disaster-recovery"
        ]
        
        for i, req in enumerate(requirements, 1):
            print(f"  {i}. {req.replace('-', ' ').title()}")
        
        print(f"  {len(requirements) + 1}. None of the above")
        
        selected = []
        while True:
            try:
                choices = input("\nEnter numbers separated by commas (e.g., 1,3,5): ").strip()
                if not choices:
                    break
                    
                choice_nums = [int(x.strip()) for x in choices.split(',')]
                
                for num in choice_nums:
                    if num == len(requirements) + 1:
                        return []
                    elif 1 <= num <= len(requirements):
                        req = requirements[num - 1]
                        if req not in selected:
                            selected.append(req)
                    else:
                        print(f"Invalid choice: {num}")
                        continue
                break
            except ValueError:
                print("Please enter valid numbers separated by commas.")
        
        return selected
    
    def _generate_recommendations(self, use_case: str, budget: str, 
                                complexity_tolerance: str, requirements: List[str]) -> Dict:
        """Generate module recommendations based on requirements"""
        
        recommendations = {}
        
        for module_type, modules in self.modules.items():
            scores = {}
            
            for module_name, module in modules.items():
                score = 0
                
                # Budget score
                if budget == "low" and module.cost_factor <= 1.0:
                    score += 3
                elif budget == "medium" and module.cost_factor <= 1.5:
                    score += 2
                elif budget == "high":
                    score += 1
                
                # Complexity score
                complexity_match = {
                    ("low", "low"): 3,
                    ("low", "medium"): 1,
                    ("low", "high"): 0,
                    ("medium", "low"): 2,
                    ("medium", "medium"): 3,
                    ("medium", "high"): 2,
                    ("high", "low"): 1,
                    ("high", "medium"): 2,
                    ("high", "high"): 3
                }
                score += complexity_match.get((complexity_tolerance, module.complexity), 0)
                
                # Use case alignment
                use_case_priorities = self.use_cases[use_case]["priorities"]
                if any(priority in module.best_for for priority in use_case_priorities):
                    score += 2
                
                # Requirements alignment
                for req in requirements:
                    if req.replace("-", "_") in module.best_for:
                        score += 1
                
                scores[module_name] = score
            
            # Select the highest scoring module
            best_module = max(scores.items(), key=lambda x: x[1])
            recommendations[module_type] = {
                "selected": best_module[0],
                "score": best_module[1],
                "alternatives": sorted(scores.items(), key=lambda x: x[1], reverse=True)[1:]
            }
        
        return recommendations
    
    def _display_recommendations(self, recommendations: Dict):
        """Display the generated recommendations"""
        print("\n🎯 Recommended Configuration")
        print("=" * 50)
        
        total_cost_factor = 0
        
        for module_type, rec in recommendations.items():
            selected_module = rec["selected"]
            module_config = self.modules[module_type][selected_module]
            
            print(f"\n📦 {module_type.replace('_', ' ').title()}: {module_config.name}")
            print(f"   {module_config.description}")
            print(f"   Cost factor: {module_config.cost_factor}x")
            print(f"   Complexity: {module_config.complexity}")
            print(f"   ✅ Pros: {', '.join(module_config.pros)}")
            if module_config.cons:
                print(f"   ⚠️  Cons: {', '.join(module_config.cons)}")
            
            total_cost_factor += module_config.cost_factor
            
            # Show alternatives
            if rec["alternatives"]:
                print(f"   🔄 Alternatives:")
                for alt_name, alt_score in rec["alternatives"][:2]:  # Show top 2 alternatives
                    alt_config = self.modules[module_type][alt_name]
                    print(f"      - {alt_config.name} (score: {alt_score})")
        
        # Cost estimation
        base_cost = 500  # Base monthly cost
        estimated_cost = base_cost * total_cost_factor
        print(f"\n💰 Estimated Monthly Cost: ${estimated_cost:.0f}")
        
        # Configuration summary
        print(f"\n📊 Configuration Summary:")
        print(f"   Total complexity: {self._calculate_overall_complexity(recommendations)}")
        print(f"   Cost optimization: {self._calculate_cost_efficiency(recommendations)}")
    
    def _calculate_overall_complexity(self, recommendations: Dict) -> str:
        complexity_scores = {"low": 1, "medium": 2, "high": 3}
        total_complexity = 0
        
        for module_type, rec in recommendations.items():
            selected_module = rec["selected"]
            module_config = self.modules[module_type][selected_module]
            total_complexity += complexity_scores[module_config.complexity]
        
        avg_complexity = total_complexity / len(recommendations)
        
        if avg_complexity <= 1.5:
            return "Low (Easy to manage)"
        elif avg_complexity <= 2.5:
            return "Medium (Moderate setup required)"
        else:
            return "High (Significant ops expertise needed)"
    
    def _calculate_cost_efficiency(self, recommendations: Dict) -> str:
        total_cost_factor = sum(
            self.modules[module_type][rec["selected"]].cost_factor
            for module_type, rec in recommendations.items()
        )
        
        if total_cost_factor <= 4:
            return "Excellent (Very cost-effective)"
        elif total_cost_factor <= 6:
            return "Good (Balanced cost/features)"
        else:
            return "Premium (Feature-rich but expensive)"
    
    def _confirm_selection(self, recommendations: Dict) -> bool:
        print(f"\n❓ Generate Terraform configuration with these modules?")
        response = input("Enter 'yes' to proceed: ").lower().strip()
        return response == 'yes'
    
    def _generate_terraform_config(self, recommendations: Dict):
        """Generate terraform tfvars configuration"""
        
        # Create configuration
        config = {
            "modules": {}
        }
        
        for module_type, rec in recommendations.items():
            config["modules"][module_type] = {
                "enabled": True,
                "type": rec["selected"]
            }
        
        # Write to file
        output_file = "environments/recommended.tfvars"
        
        # Read template
        template_file = "environments/dev.tfvars"  # Use dev as template
        
        try:
            with open(template_file, 'r') as f:
                template_content = f.read()
            
            # Replace modules section
            lines = template_content.split('\n')
            new_lines = []
            in_modules = False
            brace_count = 0
            
            for line in lines:
                if line.strip().startswith('modules = {'):
                    in_modules = True
                    new_lines.append('modules = {')
                    # Add our module configuration
                    for module_type, module_config in config["modules"].items():
                        new_lines.append(f'  {module_type} = {{')
                        new_lines.append(f'    enabled = {str(module_config["enabled"]).lower()}')
                        new_lines.append(f'    type    = "{module_config["type"]}"')
                        new_lines.append('  }')
                    continue
                
                if in_modules:
                    if '{' in line:
                        brace_count += line.count('{')
                    if '}' in line:
                        brace_count -= line.count('}')
                        if brace_count <= 0:
                            new_lines.append('}')
                            in_modules = False
                    continue
                
                new_lines.append(line)
            
            # Write new configuration
            with open(output_file, 'w') as f:
                f.write('\n'.join(new_lines))
            
            print(f"\n✅ Configuration saved to: {output_file}")
            print(f"\n🚀 To deploy this configuration:")
            print(f"   ./deploy.sh --environment recommended --action plan")
            print(f"   ./deploy.sh --environment recommended --action apply")
            
        except FileNotFoundError:
            print(f"❌ Template file not found: {template_file}")
        except Exception as e:
            print(f"❌ Error generating configuration: {e}")

def main():
    parser = argparse.ArgumentParser(description="MLOps Module Selector")
    parser.add_argument("--interactive", action="store_true", 
                       help="Run interactive selection wizard")
    parser.add_argument("--list-modules", action="store_true",
                       help="List all available modules")
    parser.add_argument("--compare", nargs=2, metavar=("MODULE_TYPE", "MODULES"),
                       help="Compare modules of a specific type")
    
    args = parser.parse_args()
    
    selector = MLOpsModuleSelector()
    
    if args.list_modules:
        selector._list_all_modules()
    elif args.compare:
        selector._compare_modules(args.compare[0], args.compare[1].split(','))
    else:
        # Default to interactive mode
        selector.interactive_selection()

if __name__ == "__main__":
    main()