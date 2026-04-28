"""Event bus instances for the factorio-assistant architecture.

Each bus has a pair:
  - An EventLinker subclass (manages subscriptions)
  - A singleton AsyncIOEventEmitter (emits events)

All snail-domain events (snail.character.*, snail.map_graph.*,
snail.screenshot.*) share a single SnailEventBus.  Events are
differentiated by their namespaced string names.

Usage:
    from assistant.event_bus import snail_events
    from assistant import events

    # Subscribe
    SnailEventBus.subscribe(
        events.SNAIL_CHARACTER_COORD_UPDATED,
        event_callback=lambda coord, source: print(f"{coord} from {source}"),
    )

    # Emit (from any component: action handlers, gRPC, CLI, etc.)
    snail_events.emit(
        events.SNAIL_CHARACTER_COORD_UPDATED,
        coord=(42, 100), source="frame",
    )
"""

from pyventus.events import AsyncIOEventEmitter, EventLinker


# ── Linkers (manage subscriptions) ─────────────────────────────────────

class SnailEventBus(EventLinker):
    """Single bus for all snail.* domain events (character, map_graph, screenshot, …).

    Events are distinguished by their namespaced string constant, e.g.
    ``"snail.character.coord_updated"`` vs ``"snail.screenshot.window"``.
    """
    __slots__ = ()


class LeafEventBus(EventLinker):
    """Subscriptions for presentation-layer events."""
    __slots__ = ()


# ── Emitters (emit events through their respective linker) ─────────────

snail_events: AsyncIOEventEmitter = AsyncIOEventEmitter(SnailEventBus)
"""Single emitter for all snail.* events."""

leaf_events: AsyncIOEventEmitter = AsyncIOEventEmitter(LeafEventBus)
"""Emitter for leaf.* events."""
