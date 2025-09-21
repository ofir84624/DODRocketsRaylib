// Vittorio Romeo (2025)

// CppCon 2025 Keynote
// "More Speed & Simplicity: Practical Data-Oriented Design in C++"
// https://www.youtube.com/watch?v=SzjJfKHygaQ

// DISCLAIMER:
// This is a RayLib implementation of my CppCon 2025 keynote demo.
//
// The original implementation uses VRSFML (my own fork of SFML) and
// can be found here:
// https://github.com/vittorioromeo/VRSFML/blob/dodtalk/examples/rockets/Rockets.cpp
//
// The rendering code in this implementation is significantly inferior
// as it does not use VRSFML's automatic batching + instanced drawing.
//
// The simulation code should be identical -- rendering should be disabled
// to get a fair comparison between OOP, AoS, and SoA.
//
// The code below is not polished or cleaned up, as it is not meant to be
// a proper RayLib demo, but just a quick port of the original VRSFML demo.

#include "imgui.h"
#include "raylib.h"
#include "raymath.h"
#include "rlImGui.h"

#include <algorithm>
#include <memory>
#include <optional>
#include <vector>


////////////////////////////////////////////////////////////
class Sampler
{
public:
    ////////////////////////////////////////////////////////////
    enum : size_t
    {
        ToIgnore = 0u
    };

    ////////////////////////////////////////////////////////////
    explicit Sampler(const size_t capacity) : m_capacity(capacity)
    {
        m_data.resize(capacity, 0.f);
    }

    ////////////////////////////////////////////////////////////
    void record(const float value)
    {
        if (m_toIgnore > 0u)
        {
            --m_toIgnore;
            return;
        }

        if (m_size < m_capacity)
        {
            // Still filling the buffer.
            m_data[m_index] = value;
            m_sum += value;
            ++m_size;
        }
        else
        {
            // Buffer is full: subtract the overwritten value and add the new one.
            m_sum           = m_sum - m_data[m_index] + value;
            m_data[m_index] = value;
        }

        // Advance index in circular fashion.
        m_index = (m_index + 1u) % m_capacity;
    }

    ////////////////////////////////////////////////////////////
    [[nodiscard]] double getAverage() const
    {
        if (m_size == 0u)
            return 0.0;

        return static_cast<double>(m_sum) / static_cast<double>(m_size);
    }

    ////////////////////////////////////////////////////////////
    [[nodiscard, gnu::always_inline]] size_t size() const
    {
        return m_size;
    }

    ////////////////////////////////////////////////////////////
    [[nodiscard, gnu::always_inline]] const float* data() const
    {
        return m_data.data();
    }

    ////////////////////////////////////////////////////////////
    void clear()
    {
        for (float& x : m_data)
            x = 0.f;

        m_size  = 0u;
        m_index = 0u;
        m_sum   = 0.f;

        m_toIgnore = ToIgnore;
    }

    ////////////////////////////////////////////////////////////
    void writeSamplesInOrder(float* target) const
    {
        if (m_size < m_capacity)
        {
            // Buffer not full: copy the valid samples (indices 0 .. m_size - 1)
            for (size_t i = 0u; i < m_size; ++i)
                target[i] = m_data[i];

            // Fill the rest with zeros.
            for (size_t i = m_size; i < m_capacity; ++i)
                target[i] = 0.f;
        }
        else
        {
            // Buffer is full: samples are stored in circular order.
            // The oldest sample is at m_data[m_index].
            size_t pos = 0u;

            // Copy from m_index to the end.
            for (size_t i = m_index; i < m_capacity; ++i)
                target[pos++] = m_data[i];

            // Then copy from the beginning up to m_index - 1.
            for (size_t i = 0u; i < m_index; ++i)
                target[pos++] = m_data[i];
        }
    }

private:
    std::vector<float> m_data;
    const size_t       m_capacity;

    size_t m_size  = 0;   // Number of valid samples currently in the buffer.
    size_t m_index = 0;   // Next index for insertion.
    float  m_sum   = 0.f; // Running sum for fast averaging.

    size_t m_toIgnore = ToIgnore;
};


////////////////////////////////////////////////////////////
/// \brief A fast pseudo-random number generator using the xoroshiro128+ algorithm.
///
/// Provides methods for generating integers, floats, vectors,
/// and random directions/points within shapes. Can be seeded.
///
/// Satisfies the C++ `UniformRandomBitGenerator` concept.
///
////////////////////////////////////////////////////////////
class [[nodiscard]] RNGFast
{
public:
    using result_type = uint64_t; //!< Type returned by `operator()` and `next()`
    using SeedType    = uint64_t; //!< Type used for seeding

private:
    ////////////////////////////////////////////////////////////
    /// \brief Rotates the bits of `x` left by `k` positions.
    ///
    /// \param x Value to rotate
    /// \param k Number of positions to rotate
    ///
    /// \return Rotated value
    ///
    ////////////////////////////////////////////////////////////
    [[nodiscard, gnu::always_inline, gnu::flatten, gnu::const]] static inline constexpr uint64_t rotl(const uint64_t x,
                                                                                                      const int k) noexcept
    {
        return (x << k) | (x >> (64 - k));
    }

    ////////////////////////////////////////////////////////////
    /// \brief Implements the SplitMix64 algorithm to initialize state.
    ///
    /// Used internally for seeding the main generator state from a single seed value.
    ///
    /// \param seed Input seed value
    ///
    /// \return A 64-bit pseudo-random number derived from the seed
    ///
    ////////////////////////////////////////////////////////////
    [[nodiscard, gnu::always_inline]] static inline uint64_t splitmix64(uint64_t& seed) noexcept
    {
        seed += 0x9e'37'79'b9'7f'4a'7c'15ULL;

        uint64_t z = seed;

        z = (z ^ (z >> 30)) * 0xbf'58'47'6d'1c'e4'e5'b9ULL;
        z = (z ^ (z >> 27)) * 0x94'd0'49'bb'13'31'11'ebULL;

        return z ^ (z >> 31);
    }

    ////////////////////////////////////////////////////////////
    /// \brief Initializes the generator state from a single seed value.
    ///
    /// \param seed The seed value to use.
    ///
    ////////////////////////////////////////////////////////////
    [[gnu::always_inline]] inline void seedInternal(SeedType seedValue) noexcept
    {
        // Use SplitMix64 to generate the initial 128-bit state
        m_state[0] = splitmix64(seedValue);
        m_state[1] = splitmix64(seedValue);

        // Ensure the initial state is not all zeros, which is invalid for xoroshiro128+
        if (m_state[0] == 0ULL && m_state[1] == 0ULL)
        {
            m_state[0] = DefaultSeed::State0; // Fallback to default non-zero state
            m_state[1] = DefaultSeed::State1;
        }
    }

    ////////////////////////////////////////////////////////////
    // Constants for the default seed if none is provided
    enum DefaultSeed : uint64_t
    {
        State0 = 123'456'789'123'456'789ULL,
        State1 = 987'654'321'987'654'321ULL
    };

    ////////////////////////////////////////////////////////////
    uint64_t m_state[2]{}; //!< Internal state of the generator

public:
    ////////////////////////////////////////////////////////////
    /// \brief Default constructor. Initializes with a fixed internal seed.
    ///
    ////////////////////////////////////////////////////////////
    explicit RNGFast() noexcept : m_state{DefaultSeed::State0, DefaultSeed::State1}
    {
    }

    ////////////////////////////////////////////////////////////
    /// \brief Constructor that initializes the generator with a specific seed.
    ///
    /// \param seed The seed value.
    ///
    ////////////////////////////////////////////////////////////
    explicit RNGFast(SeedType seed) noexcept
    {
        seedInternal(seed);
    }

    ////////////////////////////////////////////////////////////
    /// \brief Generates the next 64-bit pseudo-random number.
    ///
    /// \return A 64-bit unsigned integer.
    ///
    /// Implements the core xoroshiro128+ algorithm step.
    ///
    ////////////////////////////////////////////////////////////
    [[nodiscard, gnu::always_inline, gnu::flatten]] inline result_type next() noexcept
    {
        const uint64_t s0 = m_state[0];
        uint64_t       s1 = m_state[1];

        const uint64_t result = s0 + s1; // The '+' part of xoroshiro128+

        s1 ^= s0;

        m_state[0] = rotl(s0, 24) ^ s1 ^ (s1 << 16);
        m_state[1] = rotl(s1, 37);

        return result;
    }

    ////////////////////////////////////////////////////////////
    /// \brief Generates the next 64-bit pseudo-random number (UniformRandomBitGenerator interface).
    ///
    /// \return A 64-bit unsigned integer.
    ///
    ////////////////////////////////////////////////////////////
    [[nodiscard, gnu::always_inline]] inline result_type operator()() noexcept
    {
        return next();
    }

    ////////////////////////////////////////////////////////////
    /// \brief Returns the minimum value potentially generated (UniformRandomBitGenerator interface).
    ///
    /// \return `0`
    ///
    ////////////////////////////////////////////////////////////
    [[nodiscard]] static constexpr result_type min() noexcept
    {
        return 0;
    }

    ////////////////////////////////////////////////////////////
    /// \brief Returns the maximum value potentially generated (UniformRandomBitGenerator interface).
    ///
    /// \return Maximum value of `result_type` (`uint64_t`)
    ///
    ////////////////////////////////////////////////////////////
    [[nodiscard]] static constexpr result_type max() noexcept
    {
        return static_cast<uint64_t>(-1);
    }

    ////////////////////////////////////////////////////////////
    /// \brief Generates a pseudo-random integer within a specified range `[min, max]`.
    ///
    /// \tparam T An integral type.
    ///
    /// \param min Minimum inclusive value.
    /// \param max Maximum inclusive value.
    ///
    /// \return A pseudo-random integer in the range `[min, max]`.
    ///
    /// \warning Uses modulo biasing, which might be unsuitable for applications
    ///          requiring perfect uniformity, especially with large ranges.
    ///
    ////////////////////////////////////////////////////////////
    template <typename T>
    [[nodiscard, gnu::always_inline, gnu::flatten]] inline T getI(const T min, const T max)
    {
        using UnsignedT = std::make_unsigned_t<T>;

        const auto unsignedMin = static_cast<UnsignedT>(min);
        const auto unsignedMax = static_cast<UnsignedT>(max);

        const auto range = static_cast<uint64_t>(unsignedMax - unsignedMin) + uint64_t{1};

        return min + static_cast<T>(next() % range);
    }

    ////////////////////////////////////////////////////////////
    /// \brief Generates a pseudo-random float within a specified range `[min, max]`.
    ///
    /// \param min Minimum inclusive value.
    /// \param max Maximum inclusive value.
    ///
    /// \return A pseudo-random float in the range `[min, max]`.
    ///
    ////////////////////////////////////////////////////////////
    [[nodiscard, gnu::always_inline, gnu::flatten]] inline float getF(const float min, const float max)
    {
        // Returns a float in the inclusive range [min, max].

        // We extract 24 random bits, which is enough to fill the 23-bit mantissa of a float,
        // and normalize by dividing by (2^24 - 1).

        const auto  randomBits = static_cast<uint32_t>(next() >> (64u - 24u));          // Extract 24 bits.
        const float normalized = static_cast<float>(randomBits) / float((1 << 24) - 1); // Normalize to [0, 1].

        return min + normalized * (max - min);
    }

    ////////////////////////////////////////////////////////////
    /// \brief Generates a random 2D vector with components in specified ranges.
    ///
    /// \param mins Vec2 containing minimum inclusive values `(x, y)`.
    /// \param maxs Vec2 containing maximum inclusive values `(x, y)`.
    ///
    /// \return A random Vector2 within the specified bounds.
    ///
    ////////////////////////////////////////////////////////////
    [[nodiscard, gnu::always_inline, gnu::flatten]] inline Vector2 getVec2f(const Vector2 mins, const Vector2 maxs)
    {
        return {getF(mins.x, maxs.x), getF(mins.y, maxs.y)};
    }

    ////////////////////////////////////////////////////////////
    /// \brief Generates a random 2D vector with components between 0 and specified maximums.
    ///
    /// \param maxs Vec2 containing maximum inclusive values `(x, y)`.
    ///
    /// \return A random Vector2 within the range `[0, maxs.x]` and `[0, maxs.y]`.
    ///
    ////////////////////////////////////////////////////////////
    [[nodiscard, gnu::always_inline, gnu::flatten]] inline Vector2 getVec2f(const Vector2 maxs)
    {
        return {getF(0.f, maxs.x), getF(0.f, maxs.y)};
    }
};


////////////////////////////////////////////////////////////
RNGFast rng;


////////////////////////////////////////////////////////////
Texture2D txSmoke;
Texture2D txFire;
Texture2D txRocket;


////////////////////////////////////////////////////////////
void DrawSprite(Texture2D texture, Vector2 position, float scale, float rotation, float opacity)
{
    Rectangle source = {0.0f, 0.0f, (float)texture.width, (float)texture.height};
    Rectangle dest   = {position.x, position.y, (float)texture.width * scale, (float)texture.height * scale};
    Vector2   origin = {(float)texture.width * scale / 2.0f, (float)texture.height * scale / 2.0f};
    Color     tint   = {255, 255, 255, (unsigned char)(opacity * 255.0f)};
    DrawTexturePro(texture, source, dest, origin, rotation - 90, tint);
}


////////////////////////////////////////////////////////////
template <typename Vector, typename Predicate>
[[gnu::always_inline]] inline constexpr size_t vectorSwapAndPopIf(Vector& vector, Predicate&& predicate)
{
    const size_t initialSize = vector.size();
    size_t       currentSize = initialSize;

    for (size_t i = currentSize; i-- > 0u;)
    {
        if (!predicate(vector[i]))
            continue;

        --currentSize;
        vector[i] = std::move(vector[currentSize]);
    }

    vector.resize(currentSize);
    return static_cast<size_t>(initialSize - currentSize);
}


////////////////////////////////////////////////////////////
enum class Mode
{
    OOP,
    AOS,
    AOSImproved,
    SOAManual,
};


////////////////////////////////////////////////////////////
namespace OOP
{
////////////////////////////////////////////////////////////
struct World;


////////////////////////////////////////////////////////////
struct Entity
{
    World* world = nullptr;

    Vector2 position;
    Vector2 velocity;
    Vector2 acceleration;

    bool alive = true;

    virtual ~Entity() = default;

    virtual void update(float dt)
    {
        position += velocity * dt;
        velocity += acceleration * dt;
    }

    virtual void draw()
    {
    }
};


////////////////////////////////////////////////////////////
struct World
{
    std::vector<std::unique_ptr<Entity>> entities;

    void cleanup()
    {
        vectorSwapAndPopIf(entities, [](const auto& entity) { return !entity->alive; });
    }

    void update(float dt)
    {
        for (size_t i = 0; i < entities.size(); ++i)
            entities[i]->update(dt);
    }

    void draw()
    {
        for (const auto& entity : entities)
            entity->draw();
    }

    template <typename T, typename... Args>
    T& addEntity(Args&&... args)
    {
        auto newEntity = std::make_unique<T>(std::forward<Args>(args)...);

        T& result = *newEntity;
        entities.emplace_back(std::move(newEntity));
        result.world = this;

        return result;
    }
};


////////////////////////////////////////////////////////////
struct Emitter : Entity
{
    float spawnTimer;
    float spawnRate;

    virtual void spawnParticle() = 0;

    void update(float dt) override
    {
        Entity::update(dt);

        spawnTimer += spawnRate * dt;

        for (; spawnTimer >= 1.f; spawnTimer -= 1.f)
            spawnParticle();
    }
};


////////////////////////////////////////////////////////////
struct Particle : Entity
{
    float scale;
    float opacity;
    float rotation;

    float scaleRate;
    float opacityChange;
    float angularVelocity;

    void update(float dt) override
    {
        Entity::update(dt);

        scale += scaleRate * dt;
        opacity += opacityChange * dt;
        rotation += angularVelocity * dt;

        alive = opacity > 0.f;
    }
};


////////////////////////////////////////////////////////////
struct SmokeParticle final : Particle
{
    void draw() override
    {
        DrawSprite(txSmoke, position, scale, rotation, opacity);
    }
};


////////////////////////////////////////////////////////////
struct FireParticle final : Particle
{
    void draw() override
    {
        DrawSprite(txFire, position, scale, rotation, opacity);
    }
};


////////////////////////////////////////////////////////////
struct SmokeEmitter final : Emitter
{
    void spawnParticle() override
    {
        auto& p = world->addEntity<SmokeParticle>();

        p.position     = position;
        p.velocity     = rng.getVec2f({-0.2f, -0.2f}, {0.2f, 0.2f}) * 0.5f;
        p.acceleration = {0.f, -0.011f};

        p.scale    = rng.getF(0.0025f, 0.0035f);
        p.opacity  = rng.getF(0.05f, 0.25f);
        p.rotation = rng.getF(0.f, 6.28f);

        p.scaleRate       = rng.getF(0.001f, 0.003f) * 2.75f;
        p.opacityChange   = -rng.getF(0.001f, 0.002f) * 3.25f;
        p.angularVelocity = rng.getF(-0.02f, 0.02f);
    }
};


////////////////////////////////////////////////////////////
struct FireEmitter final : Emitter
{
    void spawnParticle() override
    {
        auto& p = world->addEntity<FireParticle>();

        p.position     = position;
        p.velocity     = rng.getVec2f({-0.3f, -0.8f}, {0.3f, -0.2f});
        p.acceleration = {0.f, 0.07f};

        p.scale    = rng.getF(0.5f, 0.7f) * 0.085f;
        p.opacity  = rng.getF(0.2f, 0.4f) * 0.85f;
        p.rotation = rng.getF(0.f, 6.28f);

        p.scaleRate       = -rng.getF(0.001f, 0.003f) * 0.25f;
        p.opacityChange   = -0.001f;
        p.angularVelocity = rng.getF(-0.002f, 0.002f);
    }
};


////////////////////////////////////////////////////////////
struct Rocket final : Entity
{
    SmokeEmitter* smokeEmitter = nullptr;
    FireEmitter*  fireEmitter  = nullptr;

    void init()
    {
        smokeEmitter               = &world->addEntity<SmokeEmitter>();
        smokeEmitter->position     = position;
        smokeEmitter->velocity     = {};
        smokeEmitter->acceleration = {};
        smokeEmitter->spawnTimer   = 0.f;
        smokeEmitter->spawnRate    = 2.5f;

        fireEmitter               = &world->addEntity<FireEmitter>();
        fireEmitter->position     = position;
        fireEmitter->velocity     = {};
        fireEmitter->acceleration = {};
        fireEmitter->spawnTimer   = 0.f;
        fireEmitter->spawnRate    = 1.25f;
    }

    void update(float dt) override
    {
        Entity::update(dt);

        smokeEmitter->position = position - Vector2{12.f, 0.f};
        fireEmitter->position  = position - Vector2{12.f, 0.f};

        if (position.x > 1680.f + 64.f)
        {
            alive = false;

            smokeEmitter->alive = false;
            fireEmitter->alive  = false;
        }
    }

    void draw() override
    {
        DrawSprite(txRocket, position, 0.15f, 90.f, 1.0f);
    }
};

} // namespace OOP


////////////////////////////////////////////////////////////
namespace AOS
{
////////////////////////////////////////////////////////////
enum class ParticleType
{
    Smoke,
    Fire
};


////////////////////////////////////////////////////////////
struct Emitter
{
    Vector2 position;
    Vector2 velocity;
    Vector2 acceleration;

    float spawnTimer;
    float spawnRate;

    ParticleType type;
};


////////////////////////////////////////////////////////////
struct Particle
{
    Vector2 position;
    Vector2 velocity;
    Vector2 acceleration;

    float scale;
    float opacity;
    float rotation;

    float scaleRate;
    float opacityChange;
    float angularVelocity;

    ParticleType type;
};


////////////////////////////////////////////////////////////
struct Rocket
{
    Vector2 position;
    Vector2 velocity;
    Vector2 acceleration;

    size_t smokeEmitterIdx;
    size_t fireEmitterIdx;
};


////////////////////////////////////////////////////////////
struct World
{
    std::vector<std::optional<Emitter>> emitters;
    std::vector<Particle>               particles;
    std::vector<Rocket>                 rockets;

    size_t addEmitter(const Emitter& emitter)
    {
        for (size_t i = 0; i < emitters.size(); ++i)
        {
            if (!emitters[i].has_value())
            {
                emitters[i].emplace(emitter);
                return i;
            }
        }
        emitters.emplace_back(emitter);
        return emitters.size() - 1;
    }

    void addRocket(const Rocket& r)
    {
        auto& rocket = rockets.emplace_back(r);

        rocket.smokeEmitterIdx = addEmitter({
            .spawnTimer = 0.f,
            .spawnRate  = 2.5f,
            .type       = ParticleType::Smoke,
        });

        rocket.fireEmitterIdx = addEmitter({
            .spawnTimer = 0.f,
            .spawnRate  = 1.25f,
            .type       = ParticleType::Fire,
        });
    }

    void update(float dt)
    {
        for (Particle& p : particles)
        {
            p.position += p.velocity * dt;
            p.velocity += p.acceleration * dt;
            p.scale += p.scaleRate * dt;
            p.opacity += p.opacityChange * dt;
            p.rotation += p.angularVelocity * dt;
        }

        for (auto& e_opt : emitters)
        {
            if (!e_opt)
                continue;

            auto& e = *e_opt;

            e.position += e.velocity * dt;
            e.velocity += e.acceleration * dt;
            e.spawnTimer += e.spawnRate * dt;

            while (e.spawnTimer >= 1.f)
            {
                e.spawnTimer -= 1.f;
                if (e.type == ParticleType::Smoke)
                {
                    particles.push_back(
                        {.position     = e.position,
                         .velocity     = rng.getVec2f({-0.2f, -0.2f}, {0.2f, 0.2f}) * 0.5f,
                         .acceleration = {0.f, -0.011f},

                         .scale    = rng.getF(0.0025f, 0.0035f),
                         .opacity  = rng.getF(0.05f, 0.25f),
                         .rotation = rng.getF(0.f, 6.28f),

                         .scaleRate       = rng.getF(0.001f, 0.003f) * 2.75f,
                         .opacityChange   = -rng.getF(0.001f, 0.002f) * 3.25f,
                         .angularVelocity = rng.getF(-0.02f, 0.02f),

                         .type = ParticleType::Smoke});
                }
                else if (e.type == ParticleType::Fire)
                {
                    particles.push_back({
                        .position     = e.position,
                        .velocity     = rng.getVec2f({-0.3f, -0.8f}, {0.3f, -0.2f}),
                        .acceleration = {0.f, 0.07f},

                        .scale    = rng.getF(0.5f, 0.7f) * 0.085f,
                        .opacity  = rng.getF(0.2f, 0.4f) * 0.85f,
                        .rotation = rng.getF(0.f, 6.28f),

                        .scaleRate       = -rng.getF(0.001f, 0.003f) * 0.25f,
                        .opacityChange   = -0.001f,
                        .angularVelocity = rng.getF(-0.002f, 0.002f),

                        .type = ParticleType::Fire,
                    });
                }
            }
        }

        for (auto& r : rockets)
        {
            r.position += r.velocity * dt;
            r.velocity += r.acceleration * dt;

            emitters[r.smokeEmitterIdx]->position = r.position - Vector2{12, 0};
            emitters[r.fireEmitterIdx]->position  = r.position - Vector2{12, 0};
        }
    }

    void cleanup()
    {
        vectorSwapAndPopIf(particles, [](const Particle& p) { return p.opacity <= 0.f; });

        vectorSwapAndPopIf(rockets,
                           [&](const Rocket& r)
        {
            if (r.position.x <= 1680.f + 64.f)
                return false;

            emitters[r.smokeEmitterIdx].reset();
            emitters[r.fireEmitterIdx].reset();

            return true; // Out of bounds
        });
    }

    void draw()
    {
        for (const auto& p : particles)
        {
            Texture2D tex = (p.type == ParticleType::Smoke) ? txSmoke : txFire;
            DrawSprite(tex, p.position, p.scale, p.rotation, p.opacity);
        }

        for (const auto& r : rockets)
            DrawSprite(txRocket, r.position, 0.15f, 90.f, 1.0f);
    }
};

} // namespace AOS

namespace AOSImproved
{
////////////////////////////////////////////////////////////
struct Emitter
{
    Vector2 position;
    Vector2 velocity;
    Vector2 acceleration;

    float spawnTimer;
    float spawnRate;
};


////////////////////////////////////////////////////////////
struct Particle
{
    Vector2 position;
    Vector2 velocity;
    Vector2 acceleration;

    float scale;
    float opacity;
    float rotation;

    float scaleRate;
    float opacityChange;
    float angularVelocity;
};


////////////////////////////////////////////////////////////
struct Rocket
{
    Vector2 position;
    Vector2 velocity;
    Vector2 acceleration;

    uint16_t smokeEmitterIdx;
    uint16_t fireEmitterIdx;
};


////////////////////////////////////////////////////////////
struct World
{
    std::vector<std::optional<Emitter>> smokeEmitters, fireEmitters;
    std::vector<Particle>               smokeParticles, fireParticles;
    std::vector<Rocket>                 rockets;

    uint16_t addEmitter(std::vector<std::optional<Emitter>>& emitterVec, const Emitter& emitter)
    {
        for (size_t i = 0; i < emitterVec.size(); ++i)
        {
            if (!emitterVec[i].has_value())
            {
                emitterVec[i].emplace(emitter);
                return i;
            }
        }
        emitterVec.emplace_back(emitter);
        return static_cast<uint16_t>(emitterVec.size() - 1);
    }

    void addRocket(const Rocket& r)
    {
        auto& rocket           = rockets.emplace_back(r);
        rocket.smokeEmitterIdx = addEmitter(smokeEmitters, {.spawnTimer = 0.f, .spawnRate = 2.5f});
        rocket.fireEmitterIdx  = addEmitter(fireEmitters, {.spawnTimer = 0.f, .spawnRate = 1.25f});
    }

    void update(float dt)
    {
        auto updateParticle = [&](Particle& p)
        {
            p.position += p.velocity * dt;
            p.velocity += p.acceleration * dt;
            p.scale += p.scaleRate * dt;
            p.opacity += p.opacityChange * dt;
            p.rotation += p.angularVelocity * dt;
        };

        auto updateEmitter = [&](auto& e, auto&& fSpawn)
        {
            if (!e.has_value())
                return;

            e->position += e->velocity * dt;
            e->velocity += e->acceleration * dt;
            e->spawnTimer += e->spawnRate * dt;

            for (; e->spawnTimer >= 1.f; e->spawnTimer -= 1.f)
                fSpawn();
        };

        for (Particle& p : smokeParticles)
            updateParticle(p);

        for (Particle& p : fireParticles)
            updateParticle(p);

        for (auto& e : smokeEmitters)
            updateEmitter(e,
                          [&]
            {
                smokeParticles.push_back({
                    .position     = e->position,
                    .velocity     = rng.getVec2f({-0.2f, -0.2f}, {0.2f, 0.2f}) * 0.5f,
                    .acceleration = {0.f, -0.011f},

                    .scale    = rng.getF(0.0025f, 0.0035f),
                    .opacity  = rng.getF(0.05f, 0.25f),
                    .rotation = rng.getF(0.f, 6.28f),

                    .scaleRate       = rng.getF(0.001f, 0.003f) * 2.75f,
                    .opacityChange   = -rng.getF(0.001f, 0.002f) * 3.25f,
                    .angularVelocity = rng.getF(-0.02f, 0.02f),
                });
            });

        for (auto& e : fireEmitters)
            updateEmitter(e,
                          [&]
            {
                fireParticles.push_back({
                    .position     = e->position,
                    .velocity     = rng.getVec2f({-0.3f, -0.8f}, {0.3f, -0.2f}),
                    .acceleration = {0.f, 0.07f},

                    .scale    = rng.getF(0.5f, 0.7f) * 0.085f,
                    .opacity  = rng.getF(0.2f, 0.4f) * 0.85f,
                    .rotation = rng.getF(0.f, 6.28f),

                    .scaleRate       = -rng.getF(0.001f, 0.003f) * 0.25f,
                    .opacityChange   = -0.001f,
                    .angularVelocity = rng.getF(-0.002f, 0.002f),
                });
            });

        for (Rocket& r : rockets)
        {
            r.position += r.velocity * dt;
            r.velocity += r.acceleration * dt;

            smokeEmitters[r.smokeEmitterIdx]->position = r.position - Vector2{12.f, 0.f};
            fireEmitters[r.fireEmitterIdx]->position   = r.position - Vector2{12.f, 0.f};
        }
    }

    void cleanup()
    {
        vectorSwapAndPopIf(smokeParticles, [](const Particle& p) { return p.opacity <= 0.f; });
        vectorSwapAndPopIf(fireParticles, [](const Particle& p) { return p.opacity <= 0.f; });

        vectorSwapAndPopIf(rockets,
                           [&](const Rocket& r)
        {
            if (r.position.x <= 1680.f + 64.f)
                return false;

            smokeEmitters[r.smokeEmitterIdx].reset();
            fireEmitters[r.fireEmitterIdx].reset();

            return true; // Out of bounds
        });
    }

    void draw()
    {
        for (const auto& p : smokeParticles)
            DrawSprite(txSmoke, p.position, p.scale, p.rotation, p.opacity);

        for (const auto& p : fireParticles)
            DrawSprite(txFire, p.position, p.scale, p.rotation, p.opacity);

        for (const auto& r : rockets)
            DrawSprite(txRocket, r.position, 0.15f, 90.f, 1.0f);
    }
};

} // namespace AOSImproved


////////////////////////////////////////////////////////////
namespace SOAManual
{
////////////////////////////////////////////////////////////
struct ParticleSoA
{
    std::vector<Vector2> positions;
    std::vector<Vector2> velocities;
    std::vector<Vector2> accelerations;

    std::vector<float> scales;
    std::vector<float> opacities;
    std::vector<float> rotations;

    std::vector<float> scaleRates;
    std::vector<float> opacityChanges;
    std::vector<float> angularVelocities;

    void forEachVector(auto&& f)
    {
        f(positions);
        f(velocities);
        f(accelerations);

        f(scales);
        f(opacities);
        f(rotations);

        f(scaleRates);
        f(opacityChanges);
        f(angularVelocities);
    }
};


////////////////////////////////////////////////////////////
using Emitter = AOSImproved::Emitter;
using Rocket  = AOSImproved::Rocket;


////////////////////////////////////////////////////////////
struct World
{
    std::vector<std::optional<Emitter>> smokeEmitters, fireEmitters;
    ParticleSoA                         smokeParticles, fireParticles;
    std::vector<Rocket>                 rockets;

    uint16_t addEmitter(std::vector<std::optional<Emitter>>& emitterVec, const Emitter& emitter)
    {
        for (size_t i = 0u; i < emitterVec.size(); ++i)
            if (!emitterVec[i].has_value())
            {
                emitterVec[i].emplace(emitter);
                return static_cast<uint16_t>(i);
            }

        emitterVec.emplace_back(emitter);
        return static_cast<uint16_t>(emitterVec.size() - 1);
    }

    void addRocket(const Rocket& r)
    {
        auto& rocket           = rockets.emplace_back(r);
        rocket.smokeEmitterIdx = addEmitter(smokeEmitters, {.spawnTimer = 0.f, .spawnRate = 2.5f});
        rocket.fireEmitterIdx  = addEmitter(fireEmitters, {.spawnTimer = 0.f, .spawnRate = 1.25f});
    }

    void update(float dt)
    {
        auto updateParticles = [&](auto& soa)
        {
            const auto nParticles = soa.positions.size();

            for (size_t i = 0u; i < nParticles; ++i)
            {
                soa.velocities[i] += soa.accelerations[i] * dt;
                soa.positions[i] += soa.velocities[i] * dt;
                soa.scales[i] += soa.scaleRates[i] * dt;
                soa.opacities[i] += soa.opacityChanges[i] * dt;
                soa.rotations[i] += soa.angularVelocities[i] * dt;
            }
        };

        updateParticles(smokeParticles);
        updateParticles(fireParticles);

        for (auto& e : smokeEmitters)
        {
            if (!e.has_value())
                continue;

            e->position += e->velocity * dt;
            e->velocity += e->acceleration * dt;
            e->spawnTimer += e->spawnRate * dt;

            for (; e->spawnTimer >= 1.f; e->spawnTimer -= 1.f)
            {
                smokeParticles.positions.push_back(e->position);
                smokeParticles.velocities.push_back(rng.getVec2f({-0.2f, -0.2f}, {0.2f, 0.2f}) * 0.5f);
                smokeParticles.accelerations.push_back({0.f, -0.011f});
                smokeParticles.scales.push_back(rng.getF(0.0025f, 0.0035f));
                smokeParticles.opacities.push_back(rng.getF(0.05f, 0.25f));
                smokeParticles.rotations.push_back(rng.getF(0.f, 6.28f));
                smokeParticles.scaleRates.push_back(rng.getF(0.001f, 0.003f) * 2.75f);
                smokeParticles.opacityChanges.push_back(-rng.getF(0.001f, 0.002f) * 3.25f);
                smokeParticles.angularVelocities.push_back(rng.getF(-0.02f, 0.02f));
            }
        }

        for (auto& e : fireEmitters)
        {
            if (!e.has_value())
                continue;

            e->position += e->velocity * dt;
            e->velocity += e->acceleration * dt;
            e->spawnTimer += e->spawnRate * dt;

            for (; e->spawnTimer >= 1.f; e->spawnTimer -= 1.f)
            {
                fireParticles.positions.push_back(e->position);
                fireParticles.velocities.push_back(rng.getVec2f({-0.3f, -0.8f}, {0.3f, -0.2f}));
                fireParticles.accelerations.push_back({0.f, 0.07f});
                fireParticles.scales.push_back(rng.getF(0.5f, 0.7f) * 0.085f);
                fireParticles.opacities.push_back(rng.getF(0.2f, 0.4f) * 0.85f);
                fireParticles.rotations.push_back(rng.getF(0.f, 6.28f));
                fireParticles.scaleRates.push_back(-rng.getF(0.001f, 0.003f) * 0.25f);
                fireParticles.opacityChanges.push_back(-0.001f);
                fireParticles.angularVelocities.push_back(rng.getF(-0.002f, 0.002f));
            }
        }

        for (Rocket& r : rockets)
        {
            r.position += r.velocity * dt;
            r.velocity += r.acceleration * dt;

            smokeEmitters[r.smokeEmitterIdx]->position = r.position - Vector2{12.f, 0.f};
            fireEmitters[r.fireEmitterIdx]->position   = r.position - Vector2{12.f, 0.f};
        }
    }

    void cleanup()
    {
        const auto soaEraseIf = [&](ParticleSoA& soa, auto&& predicate)
        {
            size_t currentSize = soa.positions.size();

            for (size_t i = currentSize; i-- > 0u;)
            {
                if (!predicate(soa, i))
                    continue;

                --currentSize;
                soa.forEachVector([&](auto& vec) { vec[i] = std::move(vec[currentSize]); });
            }

            soa.forEachVector([&](auto& vec) { vec.resize(currentSize); });
        };

        soaEraseIf(smokeParticles, [](const ParticleSoA& soa, const size_t i) { return soa.opacities[i] <= 0.f; });
        soaEraseIf(fireParticles, [](const ParticleSoA& soa, const size_t i) { return soa.opacities[i] <= 0.f; });

        vectorSwapAndPopIf(rockets,
                           [&](const Rocket& r)
        {
            if (r.position.x <= 1680.f + 64.f)
                return false;

            smokeEmitters[r.smokeEmitterIdx].reset();
            fireEmitters[r.fireEmitterIdx].reset();

            return true; // Out of bounds
        });
    }

    void draw()
    {
        const size_t nSmokeParticles = smokeParticles.positions.size();
        const size_t nFireParticles  = fireParticles.positions.size();

        for (size_t i = 0; i < nSmokeParticles; ++i)
            DrawSprite(txSmoke,
                       smokeParticles.positions[i],
                       smokeParticles.scales[i],
                       smokeParticles.rotations[i],
                       smokeParticles.opacities[i]);

        for (size_t i = 0; i < nFireParticles; ++i)
            DrawSprite(txFire,
                       fireParticles.positions[i],
                       fireParticles.scales[i],
                       fireParticles.rotations[i],
                       fireParticles.opacities[i]);

        for (const auto& r : rockets)
            DrawSprite(txRocket, r.position, 0.15f, 90.f, 1.0f);
    }
};

} // namespace SOAManual


int main()
{
    // --- Initialization ---
    InitWindow(1280, 720, "Raylib Rockets Benchmark");
    SetTargetFPS(144);
    rlImGuiSetup(true);

    txSmoke  = LoadTexture("resources/pSmoke.png");
    txFire   = LoadTexture("resources/pFire.png");
    txRocket = LoadTexture("resources/rocket.png");

    bool enableRendering = true;

    // --- Simulation State ---
    Mode  currentMode      = Mode::OOP;
    float rocketSpawnTimer = 0.0f;
    float rocketSpawnRate  = 1.0f; // rockets per second
    float simulationSpeed  = 1.0f;

    OOP::World         oopWorld;
    AOS::World         aosWorld;
    AOSImproved::World aosImprovedWorld;
    SOAManual::World   soaManualWorld;

    auto resetWorlds = [&]()
    {
        oopWorld         = {};
        aosWorld         = {};
        aosImprovedWorld = {};
        soaManualWorld   = {};
    };

    Sampler samplesUpdateMs(/* capacity */ 128u);
    Sampler samplesDrawMs(/* capacity */ 128u);

    // --- Main Loop ---
    while (!WindowShouldClose())
    {
        float dt = simulationSpeed;

        // --- Update Step ---
        const auto timeUpdateStart = GetTime();
        rocketSpawnTimer += rocketSpawnRate * simulationSpeed;

        size_t nRocketsToSpawn = 0u;

        while (rocketSpawnTimer >= 1.f)
        {
            ++nRocketsToSpawn;
            rocketSpawnTimer -= 1.f;
        }

        switch (currentMode)
        {
            case Mode::OOP:
                for (size_t i = 0; i < nRocketsToSpawn; ++i)
                {
                    auto& r        = oopWorld.addEntity<OOP::Rocket>();
                    r.position     = rng.getVec2f({-500.f, 0.f}, {-100.f, 1050.f});
                    r.velocity     = {};
                    r.acceleration = {rng.getF(0.01f, 0.025f), 0.f};
                    r.init();
                }
                oopWorld.update(dt);
                oopWorld.cleanup();
                break;
            case Mode::AOS:
                for (size_t i = 0; i < nRocketsToSpawn; ++i)
                {
                    aosWorld.addRocket({
                        .position     = rng.getVec2f({-500.f, 0.f}, {-100.f, 1050.f}),
                        .velocity     = {},
                        .acceleration = {rng.getF(0.01f, 0.025f), 0.f},
                    });
                }
                aosWorld.update(dt);
                aosWorld.cleanup();
                break;
            case Mode::AOSImproved:
                for (size_t i = 0; i < nRocketsToSpawn; ++i)
                {
                    aosImprovedWorld.addRocket({
                        .position     = rng.getVec2f({-500.f, 0.f}, {-100.f, 1050.f}),
                        .velocity     = {},
                        .acceleration = {rng.getF(0.01f, 0.025f), 0.f},
                    });
                }
                aosImprovedWorld.update(dt);
                aosImprovedWorld.cleanup();
                break;
            case Mode::SOAManual:
                for (size_t i = 0; i < nRocketsToSpawn; ++i)
                {
                    soaManualWorld.addRocket({
                        .position     = rng.getVec2f({-500.f, 0.f}, {-100.f, 1050.f}),
                        .velocity     = {},
                        .acceleration = {rng.getF(0.01f, 0.025f), 0.f},
                    });
                }
                soaManualWorld.update(dt);
                soaManualWorld.cleanup();
                break;
        }
        samplesUpdateMs.record((GetTime() - timeUpdateStart) * 1000.f);

        // --- Draw Step ---
        const auto timeDrawStart = GetTime();
        BeginDrawing();
        ClearBackground(BLACK);

        if (enableRendering)
            switch (currentMode)
            {
                case Mode::OOP:
                    oopWorld.draw();
                    break;
                case Mode::AOS:
                    aosWorld.draw();
                    break;
                case Mode::AOSImproved:
                    aosImprovedWorld.draw();
                    break;
                case Mode::SOAManual:
                    soaManualWorld.draw();
                    break;
            }

        // --- ImGui UI ---
        rlImGuiBegin();
        ImGui::SetNextWindowPos({10, 10}, ImGuiCond_Always);
        ImGui::SetNextWindowSize({350, 300}, ImGuiCond_Always);
        ImGui::Begin("Controls", nullptr, ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoCollapse);

        ImGui::Text("FPS: %d", GetFPS());
        ImGui::Text("Update Time (ms): %.3f", samplesUpdateMs.getAverage());
        ImGui::Text("Draw Time (ms): %.3f", samplesDrawMs.getAverage());

        size_t entityCount = 0;
        switch (currentMode)
        {
            case Mode::OOP:
                entityCount = oopWorld.entities.size();
                break;
            case Mode::AOS:
                entityCount = aosWorld.rockets.size() + aosWorld.emitters.size() + aosWorld.particles.size();
                break;
            case Mode::AOSImproved:
                entityCount = aosImprovedWorld.rockets.size() + aosImprovedWorld.smokeEmitters.size() +
                              aosImprovedWorld.fireEmitters.size() + aosImprovedWorld.smokeParticles.size() +
                              aosImprovedWorld.fireParticles.size();
                break;
            case Mode::SOAManual:
                entityCount = soaManualWorld.rockets.size() + soaManualWorld.smokeEmitters.size() +
                              soaManualWorld.fireEmitters.size() + soaManualWorld.smokeParticles.positions.size() +
                              soaManualWorld.fireParticles.positions.size();
                break;
        }
        ImGui::Text("Entity Count: %zu", entityCount);
        ImGui::Separator();

        if (ImGui::RadioButton("OOP", currentMode == Mode::OOP))
        {
            currentMode = Mode::OOP;
            resetWorlds();
        }
        if (ImGui::RadioButton("AoS", currentMode == Mode::AOS))
        {
            currentMode = Mode::AOS;
            resetWorlds();
        }
        if (ImGui::RadioButton("AoS (Improved)", currentMode == Mode::AOSImproved))
        {
            currentMode = Mode::AOSImproved;
            resetWorlds();
        }
        if (ImGui::RadioButton("SoA (Manual)", currentMode == Mode::SOAManual))
        {
            currentMode = Mode::SOAManual;
            resetWorlds();
        }
        ImGui::Separator();

        ImGui::SliderFloat("Sim Speed", &simulationSpeed, 0.1f, 5.0f);
        ImGui::SliderFloat("Spawn Rate", &rocketSpawnRate, 0.0f, 5.0f);

        ImGui::Checkbox("Enable Rendering", &enableRendering);

        if (ImGui::Button("Reset Simulation"))
            resetWorlds();

        ImGui::End();
        rlImGuiEnd();

        EndDrawing();
        samplesDrawMs.record((GetTime() - timeDrawStart) * 1000.f);
    }

    // --- Cleanup ---
    UnloadTexture(txSmoke);
    UnloadTexture(txFire);
    UnloadTexture(txRocket);
    rlImGuiShutdown();
    CloseWindow();
    return 0;
}
