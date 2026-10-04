"""Bounded exact pair-indexed observation reads; no selection or interpretation."""

from dataclasses import dataclass


@dataclass(frozen=True)
class PairObservationHistory:
    groups: tuple

    @classmethod
    def from_groups(cls, groups):
        if len(groups) > 64:
            raise ValueError("pair history group bound exceeded")
        seen = set()
        normalized = []
        for group in groups:
            key, rows = tuple(group["group_key"]), tuple(group["rows"])
            if len(key) != 2 or len(set(key)) != 2 or key in seen:
                raise ValueError("invalid or duplicate pair history group")
            if len(rows) > 64:
                raise ValueError("pair history row bound exceeded")
            if any(tuple(row["entity_refs"]) != key for row in rows):
                raise ValueError("pair history row belongs to another group")
            seen.add(key)
            normalized.append((key, rows))
        return cls(tuple(sorted(normalized)))

    @classmethod
    def from_rows(cls, rows):
        if isinstance(rows, cls):
            return rows
        # Legacy flat callers retain their original bound.
        if len(rows) > 64:
            raise ValueError("transformation pair history bound exceeded")
        groups = {}
        for row in rows:
            groups.setdefault(tuple(row["entity_refs"]), []).append(row)
        return cls.from_groups(tuple({"group_key": key, "rows": values}
                                     for key, values in sorted(groups.items())))

    def iter_rows(self):
        for _, rows in self.groups:
            yield from rows

    def rows_for(self, pair):
        pair = tuple(pair)
        return next((rows for key, rows in self.groups if key == pair), ())
