"""Pure bounded multiset rewriting and distinct input-allocation witnesses.

The caller declares signature substitutability, templates, their order, and
which observations are rebound. No spatial route or action is selected here.
"""
from __future__ import annotations

from collections import Counter, deque

from pydantic import Field, model_validator

from agents.yf_arc3_v5.logos.types import FrozenModel, Ref


class SignatureStockItem(FrozenModel):
    item_ref: Ref
    signature_ref: Ref


class ConsumptiveTemplate(FrozenModel):
    template_ref: Ref
    input_signature_refs: tuple[Ref, ...] = Field(min_length=1, max_length=8)
    output_signature_ref: Ref


class ConsumptiveDependencyRequest(FrozenModel):
    schema_version: Ref = "yf_arc3_v5.consumptive_dependency_request.v1"
    stock: tuple[SignatureStockItem, ...] = Field(max_length=64)
    templates: tuple[ConsumptiveTemplate, ...] = Field(max_length=32)
    required_signature_refs: tuple[Ref, ...] = Field(min_length=1, max_length=16)
    observed_template_refs: tuple[Ref, ...] = Field(max_length=32)
    signature_substitution_declared: bool
    virtual_output_prefix: Ref
    maximum_states: int = Field(ge=1, le=4096)
    maximum_depth: int = Field(ge=0, le=64)

    @model_validator(mode="after")
    def validate_references(self) -> "ConsumptiveDependencyRequest":
        for refs in (
            tuple(item.item_ref for item in self.stock),
            tuple(template.template_ref for template in self.templates),
            self.observed_template_refs,
        ):
            if len(set(refs)) != len(refs):
                raise ValueError("stock, templates and observations require unique references")
        if not set(self.observed_template_refs) <= {t.template_ref for t in self.templates}:
            raise ValueError("an observation must refer to a supplied template")
        if any(item.item_ref == f"{self.virtual_output_prefix}:{i}" for item in self.stock for i in range(self.maximum_depth)):
            raise ValueError("virtual outputs must not alias current stock")
        return self


class ConsumptiveApplication(FrozenModel):
    template_ref: Ref
    input_item_refs: tuple[Ref, ...]
    output_item_ref: Ref
    output_signature_ref: Ref
    input_leaf_refs: tuple[Ref, ...]


class ConsumptiveDependencyMeasurements(FrozenModel):
    schema_version: Ref = "yf_arc3_v5.consumptive_dependency_measurements.v1"
    enumeration_complete: bool
    expanded_state_count: int = Field(ge=0)
    witness_present: bool
    applications: tuple[ConsumptiveApplication, ...] = Field(max_length=64)
    final_stock: tuple[SignatureStockItem, ...] = Field(max_length=64)
    unobserved_template_refs: tuple[Ref, ...] = Field(max_length=32)
    first_template_observed: bool | None


def _instantiate(request: ConsumptiveDependencyRequest, route: tuple[int, ...]):
    available=list(request.stock)
    leaves={item.item_ref:(item.item_ref,) for item in available}
    applications=[]
    for ordinal,index in enumerate(route):
        template=request.templates[index]
        selected=[]
        for signature in template.input_signature_refs:
            item=next(item for item in available if item.signature_ref==signature and item.item_ref not in selected)
            selected.append(item.item_ref)
        input_leaves=tuple(ref for item in selected for ref in leaves[item])
        if len(set(input_leaves)) != len(input_leaves):
            raise ValueError("consumptive branches cannot reuse an initial stock identity")
        output_ref=f"{request.virtual_output_prefix}:{ordinal}"
        available=[item for item in available if item.item_ref not in selected]
        available.append(SignatureStockItem(item_ref=output_ref,signature_ref=template.output_signature_ref))
        leaves[output_ref]=input_leaves
        applications.append(ConsumptiveApplication(template_ref=template.template_ref,
            input_item_refs=tuple(selected),output_item_ref=output_ref,
            output_signature_ref=template.output_signature_ref,input_leaf_refs=input_leaves))
    return tuple(applications),tuple(available)


def measure_consumptive_dependency(
    request: ConsumptiveDependencyRequest,
) -> ConsumptiveDependencyMeasurements:
    """Return one finite symbolic witness under supplied template order.

    Complete without a witness means only that the supplied finite multiset
    model has no witness. Unknown transitions remain explicitly identified.
    Current body bindings and spatial feasibility must be remeasured separately.
    """
    def result(complete,expanded,route=None):
        applications,final_stock=_instantiate(request,route) if route is not None else ((),request.stock)
        used=tuple(dict.fromkeys(request.templates[i].template_ref for i in (route or ())))
        unobserved=tuple(ref for ref in used if ref not in request.observed_template_refs)
        first=(request.templates[route[0]].template_ref in request.observed_template_refs) if route else None
        return ConsumptiveDependencyMeasurements(enumeration_complete=complete,expanded_state_count=expanded,
            witness_present=route is not None,applications=applications,final_stock=final_stock,
            unobserved_template_refs=unobserved,first_template_observed=first)
    if not request.signature_substitution_declared:
        return result(False,0)
    signatures=tuple(sorted({item.signature_ref for item in request.stock}
        | set(request.required_signature_refs)
        | {signature for template in request.templates for signature in (*template.input_signature_refs,template.output_signature_ref)}))
    indexes={signature:i for i,signature in enumerate(signatures)}
    counts=Counter(item.signature_ref for item in request.stock)
    initial=tuple(counts[signature] for signature in signatures)
    demand=Counter(request.required_signature_refs)
    requirements=tuple(demand[signature] for signature in signatures)
    rows=tuple((Counter(template.input_signature_refs),template.output_signature_ref) for template in request.templates)
    queue=deque([(initial,())]); seen={initial}; expanded=0; depth_pruned=False
    while queue:
        current,route=queue.popleft(); expanded+=1
        if all(count>=required for count,required in zip(current,requirements)):
            return result(True,expanded,route)
        for index,(inputs,output) in enumerate(rows):
            if any(current[indexes[signature]]<count for signature,count in sorted(inputs.items())):
                continue
            next_counts=list(current)
            for signature,count in sorted(inputs.items()):
                next_counts[indexes[signature]]-=count
            next_counts[indexes[output]]+=1
            next_state=tuple(next_counts)
            if next_state in seen:
                continue
            if len(route)>=request.maximum_depth:
                depth_pruned=True
                continue
            if len(seen)>=request.maximum_states:
                return result(False,expanded)
            seen.add(next_state)
            queue.append((next_state,(*route,index)))
    return result(not depth_pruned,expanded)
